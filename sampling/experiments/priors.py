import os
import argparse
from functools import partial

import logsumexpv2 as lse
import torch

from sampling.samplers import (
    mala_sampler,
    gibbs_sampler,
    laplace_latent_sampler,
    zero_map,
    reciprocal_map,
    student_t_latent_sampler,
    identity_map,
    construct_adj_conv,
    timed_collector,
    potential,
    mu_map_normal,
    var_map_normal,
    mu_map_gmm,
    var_map_gmm,
    gmm_lat_sampler,
)

from sampling.util import (
    rng_seed,
    get_data_dir,
    construct_matrix_from_operator,
)


# Generates MALA and Gibbs samples from a grid graph model
def sample_grid_model(
    pot,
    tau,
    gibbs_config,
    num_chains,
    num_iters_mala,
    num_iters_gibbs,
    patch_size,
    dtype,
    device,
):
    # Basic sanity checks
    assert tau > 0.0
    assert num_chains > 0
    assert num_iters_mala >= 0
    assert num_iters_gibbs >= 0
    assert patch_size > 0

    # Set up convolutional layer that contains all MRF filters
    conv = torch.nn.Conv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=(3, 3),
        bias=False,
        dtype=dtype,
        padding_mode="circular",
        padding="same",
    )
    conv.weight.requires_grad = False

    # Initialize the kernel weights
    conv.weight.data[:, :, :, :] = 0.0

    # Horizontal
    conv.weight.data[0, 0, 1, 1] = -1.0
    conv.weight.data[0, 0, 1, 2] = 1.0

    # Vertical
    conv.weight.data[1, 0, 1, 1] = -1.0
    conv.weight.data[1, 0, 2, 1] = 1.0

    # Construct the adjoint
    conv.to(device)
    adj_conv = construct_adj_conv(conv)

    # Construct the adjoint squared required for preconditioning
    adj_conv_sqr = construct_adj_conv(conv)
    with torch.no_grad():
        adj_conv_sqr.weight.data = adj_conv_sqr.weight.data**2

    # Construct the necessary functions for MALA
    def energy(x):
        batch_size = x.shape[0]

        # Compute the model and tie-breaking energies
        model_energy = torch.sum(pot(conv(x)).view(batch_size, -1), dim=1)
        tie_breaking_energy = torch.sum(x.view(batch_size, -1), dim=1) ** 2

        return model_energy + tie_breaking_energy

    def grad_energy(x):
        # Clone the argument
        x = torch.clone(x)
        x.requires_grad = True

        # Backprop
        loss = torch.sum(energy(x))
        loss.backward()

        return x.grad.detach()

    def log_p(x):
        return -energy(x)

    def grad_log_p(x):
        return -grad_energy(x)

    # Define the initial value
    x_init = torch.zeros(
        (num_chains, 1, patch_size, patch_size), dtype=dtype, device=device
    )

    # Construct callbacks to collect the marginals samples
    f_mala = torch.zeros((num_iters_mala + 1, num_chains), dtype=dtype, device=device)
    f_gibbs = torch.zeros((num_iters_gibbs + 1, num_chains), dtype=dtype, device=device)

    # Estimate the worst-case direction
    D = construct_matrix_from_operator(conv, x_init[0])
    Lambda = D.t() @ D
    _, eigvectors = torch.linalg.eigh(Lambda)
    f = eigvectors[:, 1].reshape((patch_size, patch_size))

    # Define transform that computs the required marginals
    def marginal_transform(x):
        return torch.sum(f[None, None, :, :] * x, dim=(1, 2, 3))

    # Construct the callbacks
    times_mala = torch.zeros((num_iters_mala + 1,), dtype=torch.int64, device=device)
    times_gibbs = torch.zeros((num_iters_gibbs + 1,), dtype=torch.int64, device=device)
    callback_mala = partial(
        timed_collector, times_mala[1:], f_mala[1:], transform=marginal_transform
    )
    callback_gibbs = partial(
        timed_collector, times_gibbs[1:], f_gibbs[1:], transform=marginal_transform
    )

    # Compute the number of iterations and run the samplers
    samples_mala = None
    if num_iters_mala > 0:
        rng_seed()
        samples_mala = mala_sampler(
            log_p, grad_log_p, tau, x_init, num_iters_mala, True, callback_mala
        )

    samples_gibbs = None
    if num_iters_gibbs > 0:
        mu_map, var_map, latent_sampler = gibbs_config
        rng_seed()
        samples_gibbs = gibbs_sampler(
            conv,
            adj_conv,
            mu_map,
            var_map,
            latent_sampler,
            x_init,
            num_iters_gibbs,
            True,
            callback_gibbs,
            adj_conv_sqr,
        )

    return f, times_mala, samples_mala, times_gibbs, samples_gibbs, f_mala, f_gibbs


# NOTE: The energy here should not be rescaled as rescaling changes the gradient, which in turn produces wrong samples
# with gradient-style samplers.
def pot_gmm_gen(weights, means, sigmas, f):
    f_act = lse.pot_act(f, weights, means, sigmas)[0]

    return f_act


# Defines the considered models
def get_configurations(dtype, device):
    # Normal configuration
    mean = torch.tensor(0.0, dtype=dtype, device=device)
    std = torch.tensor(0.4, dtype=dtype, device=device)
    normal = torch.distributions.Normal(mean, std)
    pot_normal = partial(potential, normal)

    normal_config = (
        partial(mu_map_normal, mean),
        partial(var_map_normal, std**2),
        lambda x: torch.ones_like(x),
    )

    # Laplace configuration
    loc = torch.tensor(0.0, dtype=dtype, device=device)
    b = torch.tensor(0.3, dtype=dtype, device=device)
    laplace = torch.distributions.Laplace(loc, b)
    pot_laplace = partial(potential, laplace)

    laplace_config = (zero_map, identity_map, partial(laplace_latent_sampler, b.item()))

    # Student-t configuration
    df = torch.tensor(6.0, dtype=dtype, device=device)
    student_t = torch.distributions.StudentT(df)
    pot_student_t = partial(potential, student_t)

    student_t_config = (zero_map, reciprocal_map, partial(student_t_latent_sampler, df))

    # GMM
    weights = torch.tensor(
        [
            [0.1058, 0.0054, 0.0886, 0.0733, 0.0667, 0.0077, 0.0591, 0.3213, 0.2721],
            [0.1058, 0.0054, 0.0886, 0.0733, 0.0667, 0.0077, 0.0591, 0.3213, 0.2721],
        ],
        dtype=dtype,
        device=device,
    )
    means = torch.zeros_like(weights)
    variances = torch.tensor(
        [
            [
                2.5000e-05,
                4.0000e-04,
                6.2500e-04,
                4.9000e-03,
                5.6250e-03,
                7.2900e-02,
                7.5625e-02,
                7.8400e-02,
                8.1225e-02,
            ],
            [
                2.5000e-05,
                4.0000e-04,
                6.2500e-04,
                4.9000e-03,
                5.6250e-03,
                7.2900e-02,
                7.5625e-02,
                7.8400e-02,
                8.1225e-02,
            ],
        ],
        dtype=dtype,
        device=device,
    )
    sigmas = torch.sqrt(variances)

    pot_gmm = partial(pot_gmm_gen, weights, means, sigmas)

    gmm_config = (
        partial(mu_map_gmm, means),
        partial(var_map_gmm, variances),
        partial(gmm_lat_sampler, weights, means, sigmas),
    )

    # Construct the configuration dicts
    mala_conf_dict = {
        "normal": pot_normal,
        "laplace": pot_laplace,
        "student-t": pot_student_t,
        "gmm": pot_gmm,
    }
    gibbs_config_dict = {
        "normal": normal_config,
        "laplace": laplace_config,
        "student-t": student_t_config,
        "gmm": gmm_config,
    }

    return mala_conf_dict, gibbs_config_dict


# Experiment runner function
def run_experiments(
    num_chains, num_iters_mala, num_iters_gibbs, patch_size, factor_filter_list
):
    # Basic sanity checks
    assert num_chains > 0
    assert num_iters_mala >= 0
    assert num_iters_gibbs >= 0
    assert patch_size > 0

    # Define the dtype and device
    dtype = torch.float64
    device = "cuda"

    # Get model configurations
    mala_conf_dict, gibbs_config_dict = get_configurations(dtype, device)

    # Define the mala step sizes
    if patch_size == 12:
        mala_step_size_dict = {
            "normal": 5e-3,
            "laplace": 2.625e-4,
            "student-t": 6e-3,
            "gmm": 3e-6,
        }
    elif patch_size == 24:
        mala_step_size_dict = {
            "normal": 1.45e-3,
            "laplace": 1.7625e-5,
            "student-t": 1.5e-3,
            "gmm": 10.5e-7,
        }
    elif patch_size == 48:
        mala_step_size_dict = {
            "normal": 3.75e-4,
            "laplace": 13e-7,
            "student-t": 3.75e-4,
            "gmm": 2.25e-7,
        }
    else:
        # 96 x 96 as default
        mala_step_size_dict = {
            "normal": 9e-5,
            "laplace": 2e-8,
            "student-t": 9e-5,
            "gmm": 1e-7,
        }

    # Get the data directory
    root_dir = get_data_dir()

    # Run the experiments
    print()
    print(80 * "*")
    print(f"Running prior sampling experiments for patch_size = {patch_size}")
    print(80 * "*")
    for pot_name, gibbs_config in gibbs_config_dict.items():
        # Skip factors that were not specified by the configuration
        if pot_name not in factor_filter_list:
            continue

        print()
        print(f"Sampling for {pot_name} factors...")
        pot, tau = mala_conf_dict[pot_name], mala_step_size_dict[pot_name]

        # Generate the samples and the marginals
        f, times_mala, samples_mala, times_gibbs, samples_gibbs, f_mala, f_gibbs = (
            sample_grid_model(
                pot,
                tau,
                gibbs_config,
                num_chains,
                num_iters_mala,
                num_iters_gibbs,
                patch_size,
                dtype,
                device,
            )
        )

        # Get the data directory of this experiment
        dir = (
            root_dir / "experiments" / "sampling" / "prior" / pot_name / f"{patch_size}"
        )

        print(dir)

        # Removing old files
        os.system(f"rm -rf {dir / '{*,.*}'}")

        # Dump worst case direction, samples and marginals to disk
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(f, dir / "f.pth")

        # Dump MALA results
        if num_iters_mala > 0:
            torch.save(samples_mala, dir / "samples_mala.pth")
            torch.save(times_mala, dir / "times_mala.pth")
            torch.save(f_mala, dir / "f_mala.pth")

        # Dump Gibbs results
        if num_iters_gibbs > 0:
            torch.save(samples_gibbs, dir / "samples_gibbs.pth")
            torch.save(times_gibbs, dir / "times_gibbs.pth")
            torch.save(f_gibbs, dir / "f_gibbs.pth")

        print("Done!")
    print(80 * "*")


# Main
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run prior sampling experiments with specified parameters."
    )
    parser.add_argument(
        "--patch_sizes",
        nargs="+",
        type=int,
        default=[12, 24, 48, 96],
        help="List of patch sizes (default: [12, 24, 48, 96])",
    )
    parser.add_argument(
        "--num_chains", type=int, default=1_000, help="Number of chains (default: 1000)"
    )
    parser.add_argument(
        "--num_iters_mala",
        type=int,
        default=3000000,
        help="Number of iterations for MALA sampling (default: 3000000)",
    )
    parser.add_argument(
        "--num_iters_gibbs",
        type=int,
        default=15000,
        help="Number of iterations for Gibbs sampling (default: 15000)",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        type=str,
        default=["normal", "laplace", "student-t", "gmm"],
        help="List of factors to apply in the prior (default: ['normal', 'laplace', 'student-t', 'gmm'])",
    )

    # Set experiment parameters
    args = parser.parse_args()
    patch_sizes = args.patch_sizes
    num_chains = args.num_chains
    num_iters_mala = args.num_iters_mala
    num_iters_gibbs = args.num_iters_gibbs
    factors = args.factors

    # Run the prior sampling experiments with specified parameters
    for patch_size in patch_sizes:
        run_experiments(
            num_chains, num_iters_mala, num_iters_gibbs, patch_size, factors
        )


if __name__ == "__main__":
    # Example run command with all arguments specified:
    # python -m sampling.experiments.priors --patch_sizes 12 24 48 96 --num_chains 1000 --num_iters_mala 3000000
    # --num_iters_gibbs 15000 --factors normal laplace student-t gmm

    # Run the experiments
    main()
