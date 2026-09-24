import argparse
import os
import time
from functools import partial
from typing import List

import torch
from tqdm import tqdm

import numpy as np
import math

from sampling.experiments.priors import (
    potential,
    mu_map_normal,
    var_map_normal,
    pot_gmm_gen,
    mu_map_gmm,
    var_map_gmm,
    gmm_lat_sampler,
)
from sampling.samplers import (
    mala_sampler,
    gibbs_sampler,
    laplace_latent_sampler,
    construct_adj_conv,
)
from sampling.samplers import (
    zero_map,
    identity_map,
    reciprocal_map,
    student_t_latent_sampler,
    timed_collector,
)
from sampling.util import rng_seed, get_data_dir, init_matplotlib


# Utility function to load the test images
def load_test_images(dtype, device):
    return torch.load("samples_bsd500.pth", weights_only=True).to(
        dtype=dtype, device=device
    )


# Defines the considered models
def get_model_configurations(dtype, device):
    # Normal configuration
    mean = torch.tensor(0.0, dtype=dtype, device=device)
    std = torch.tensor(0.4, dtype=dtype, device=device)
    normal = torch.distributions.Normal(mean, std)
    pot_normal = partial(potential, normal)

    normal_config = (
        partial(mu_map_normal, mean),
        partial(var_map_normal, std ** 2),
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

    # Define the scalings for the potentials to match the value range
    scalings = {
        "normal": 3.6545774260173203,
        "laplace": 3.7761423265910774,
        "student-t": 13.117132132132133,
        "gmm": 0.5803,
    }

    return mala_conf_dict, gibbs_config_dict, scalings, std


# Define the utility function to store the conditional mean and std across the iterations
def marginal_transform(item):
    return torch.cat((item.mean(dim=0, keepdim=True), item.std(dim=0, keepdim=True)), dim=0)


def run_denoising_experiments(
        num_chains: int,
        num_iters_mala: int,
        num_iters_gibbs: int,
        factor_filter_list: List[str],
        num_test_images: int,
        std: float,
        preconditioning: bool = False,
        posterior_scales: bool = False,
        dtype=torch.float64,
        device="cuda",
        timed_run: bool = False,
):
    # Basic sanity checks
    assert num_chains > 0
    assert num_iters_mala >= 0
    assert num_iters_gibbs >= 0
    assert 0 < num_test_images <= 1000
    assert std > 0
    assert preconditioning in [True, False]

    # Load the test images
    x_gt_tensor = load_test_images(dtype, device)

    # Fix random seed for repeatability
    rng_seed()

    # Construct their noisy versions
    b_tensor = x_gt_tensor + std * torch.randn_like(x_gt_tensor)

    # Select the specified amount of test images
    x_gt_tensor = x_gt_tensor[:num_test_images]
    b_tensor = b_tensor[:num_test_images]

    # Define the priors
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
        adj_conv_sqr.weight.data = adj_conv_sqr.weight.data ** 2

    # Get the model configurations
    mala_conf_dict, gibbs_config_dict, scalings, std_normal = get_model_configurations(
        dtype, device
    )

    if posterior_scales:
        scalings = {
            "normal": 3.7355,
            "laplace": 4.145200000000001,
            "student-t": 13.9207,
            "gmm": 0.8334,
        }

    # NOTE: MALA is not used in the experiments and therefore these step-sizes are not properly tuned
    # Define the mala step sizes
    mala_step_size_dict = {
        "normal": 5.5e-5,
        "laplace": 3e-8,
        "student-t": 6e-5,
        "gmm": 4.75e-6,
    }

    # Get the data directory
    root_dir = get_data_dir()

    # Run the experiments
    print()
    print(80 * "*")
    print("Running posterior sampling experiments for denoising")
    print(80 * "*")
    for pot_name, gibbs_config in gibbs_config_dict.items():
        # Skip factors that were not specified by the configuration
        if pot_name not in factor_filter_list:
            continue

        print("\n")
        print(f"Sampling for {pot_name} factors...")
        pot, tau, scaling = (
            mala_conf_dict[pot_name],
            mala_step_size_dict[pot_name],
            scalings[pot_name],
        )

        # Get the data directory of this experiment
        if posterior_scales:
            dir = (
                    root_dir
                    / "experiments"
                    / "sampling"
                    / "posterior"
                    / "denoising-posterior-scales"
                    / pot_name
            )
        else:
            dir = (
                    root_dir
                    / "experiments"
                    / "sampling"
                    / "posterior"
                    / "denoising"
                    / pot_name
            )

        # Removing old files
        os.system(f"rm -rf {dir / '{*,.*}'}")

        # Create directory if it does not exist already
        if not os.path.exists(dir):
            os.makedirs(dir)

        # Dump the samples and the input
        torch.save(x_gt_tensor, dir / "ground_truth.pth")
        torch.save(b_tensor, dir / "noisy_inputs.pth")

        for i in tqdm(range(num_test_images)):
            # Extract the current input
            b = b_tensor[i].clone()

            # Construct the necessary functions for MALA
            def log_p(x):
                # Extract the batch-size
                batch_size = x.shape[0]

                # Construct the energy
                R = torch.sum((pot(conv(scaling * x))).view(batch_size, -1), dim=1)
                return (
                        -0.5
                        * torch.sum((x - b[None]).view(batch_size, -1) ** 2, dim=1)
                        / (std ** 2)
                        - R
                )

            def grad_log_p(x):
                # Clone the argument
                x = torch.clone(x)
                x.requires_grad = True

                # Backprop
                loss = torch.sum(log_p(x))
                loss.backward()

                return x.grad.detach()

            # Construct the MALA callbacks
            x_init_mala = torch.zeros(
                (num_chains, *b.shape), dtype=b.dtype, device=b.device
            )

            if timed_run:
                marginals_mala = torch.zeros((num_iters_mala + 1, 2, *b.shape), dtype=b.dtype, device=b.device)
                times_mala = torch.zeros((num_iters_mala + 1,), dtype=torch.int64, device=device)

                callback_mala = partial(
                    timed_collector, times_mala[1:], marginals_mala[1:], transform=marginal_transform
                )

            # Run MALA sampling
            if num_iters_mala > 0:
                if timed_run:
                    times_mala[0] = time.perf_counter_ns()
                    samples_mala = mala_sampler(log_p, grad_log_p, tau, x_init_mala, num_iters_mala, False,
                                                callback_mala)
                    torch.save(marginals_mala, f"{dir}/{i:06}-marginals-mala.pth")
                    torch.save(times_mala, f"{dir}/{i:06}-times-mala.pth")
                else:
                    samples_mala = mala_sampler(
                        log_p, grad_log_p, tau, x_init_mala, num_iters_mala, False
                    )
                torch.save(samples_mala, f"{dir}/{i:06}-mala.pth")

            # Construct the necessary functions for Gibbs
            def K(x):
                with torch.no_grad():
                    return torch.concat((x.clone(), conv(scaling * x)), dim=1)

            def adj_K(y):
                with torch.no_grad():
                    y_data = y[:, :1, :, :].clone()
                    y_prior = y[:, 1:, :, :]
                    return y_data + adj_conv(scaling * y_prior)

            mu_map, var_map, latent_sampler = gibbs_config

            def mu_map_denoising(z):
                mean = torch.zeros(z.shape, device=device, dtype=dtype)
                mean[:, 0, :, :] = b
                mean[:, 1:, :, :] = mu_map(z[:, 1:, :, :])

                return mean

            def var_map_denoising(z):
                variances = torch.zeros(z.shape, device=device, dtype=dtype)
                variances[:, 0, :, :] = std ** 2
                variances[:, 1:, :, :] = var_map(z[:, 1:, :, :])

                return variances

            # Handle the datatype properly for the GMM factors
            if pot_name == "gmm":

                def latent_sampler_denoising(u):
                    z = torch.ones(u.shape, device=device, dtype=torch.int64)
                    z[:, 1:, :, :] = latent_sampler(u[:, 1:, :, :])

                    return z

            else:

                def latent_sampler_denoising(u):
                    z = torch.ones_like(u)
                    z[:, 1:, :, :] = latent_sampler(u[:, 1:, :, :])

                    return z

            # Preconditioner if used
            def adj_K_sqr(inv_var):
                return inv_var[:, :1, :, :] + (scaling ** 2) * adj_conv_sqr(
                    inv_var[:, 1:, :, :]
                )

            preconditioner = adj_K_sqr if preconditioning else None

            # Run Gibbs sampling
            x_init_gibbs = torch.zeros(
                (num_chains, *b.shape), dtype=b.dtype, device=b.device
            )

            num_iterations_gibbs = 1 if pot_name == "normal" else num_iters_gibbs
            if timed_run:
                marginals_gibbs = torch.zeros((num_iterations_gibbs + 1, 2, *b.shape), dtype=b.dtype, device=b.device)
                times_gibbs = torch.zeros((num_iterations_gibbs + 1,), dtype=torch.int64, device=device)

                callback_gibbs = partial(
                    timed_collector, times_gibbs[1:], marginals_gibbs[1:], transform=marginal_transform
                )

            if num_iters_gibbs > 0:
                if timed_run:
                    times_gibbs[0] = time.perf_counter_ns()
                    samples_gibbs = gibbs_sampler(
                        K,
                        adj_K,
                        mu_map_denoising,
                        var_map_denoising,
                        latent_sampler_denoising,
                        x_init_gibbs,
                        num_iterations_gibbs,
                        False,
                        adj_K_sqr=preconditioner,
                        verbose=False,
                        callback=callback_gibbs,
                    )
                    torch.save(marginals_gibbs, f"{dir}/{i:06}-marginals-gibbs.pth")
                    torch.save(times_gibbs, f"{dir}/{i:06}-times-gibbs.pth")
                else:
                    samples_gibbs = gibbs_sampler(
                        K,
                        adj_K,
                        mu_map_denoising,
                        var_map_denoising,
                        latent_sampler_denoising,
                        x_init_gibbs,
                        num_iterations_gibbs,
                        False,
                        adj_K_sqr=preconditioner,
                        verbose=False
                    )
                torch.save(samples_gibbs, f"{dir}/{i:06}-gibbs.pth")

        print("Done!")
    print(80 * "*")


# Define batch versions of the forward and adjoint DCT operators
def dct_(x_):
    sh = x_.shape
    N = sh[-1]
    x = x_.contiguous().view(-1, sh[-1])
    temp = torch.hstack((x[:, ::2], torch.flip(x, (-1,))[:, N % 2:: 2]))
    temp = torch.fft.fft(temp)
    k = torch.exp(-1j * np.pi * torch.arange(N).to(x_) / (2 * N))
    X = (temp * k).real
    X[:, 0] /= math.sqrt(2)
    return (X * math.sqrt(2 / N)).view(sh)


def dct(x):
    return dct_(dct_(x).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


def dct_adj_(x_):
    sh = x_.shape
    N = sh[-1]
    x = x_.contiguous().view(-1, N)
    factor = -1j * np.pi / (N * 2)
    temp = x * torch.exp(torch.arange(N).to(x_) * factor)[None]
    temp[:, 0] /= math.sqrt(2)
    temp = torch.fft.fft(temp).real
    result = torch.empty_like(x)
    result[:, ::2] = temp[:, : (N + 1) // 2]
    indices = torch.arange(-1 - N % 2, -N, -2)
    result[:, indices] = temp[:, (N + 1) // 2:]
    return result.view(sh) / math.sqrt(N / 2)


def dct_adj(x):
    return dct_adj_(dct_adj_(x).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


def run_dct_inpainting_experiments(
        num_chains: int,
        num_iters_mala: int,
        num_iters_gibbs: int,
        factor_filter_list: List[str],
        num_test_images: int,
        preconditioning: bool = False,
        posterior_scales: bool = False,
        dtype=torch.float64,
        device="cuda",
        timed_run: bool = False,
):
    # Basic sanity checks
    assert num_chains > 0
    assert num_iters_mala >= 0
    assert num_iters_gibbs >= 0
    assert 0 < num_test_images <= 1000
    assert preconditioning in [True, False]

    # Load the test images
    x_gt_tensor = load_test_images(dtype, device)

    # Fix random seed for repeatability
    rng_seed()

    # Construct the measurements
    # Probability of zeroing out elements
    p = 0.25

    # Create a mask with the same shape as the input tensors and convert it to bool
    mask_tensor = torch.bernoulli((1 - p) * torch.ones_like(x_gt_tensor)).bool()

    # Set the mask for the upper left 32 x 32 part to True in every input image, since we want to keep the low
    # frequency components
    mask_tensor[:, :, :32, :32] = True

    # Compute the DCT transform
    X_tensor = dct(x_gt_tensor)

    # Add iid Gaussian noise
    std = 0.1
    X_tensor += std * torch.randn_like(X_tensor)

    # Zero out some the selected elements
    X_tensor[~mask_tensor] = 0.0

    # Compute the zero-fill solution via the inverse DCT
    x_zero_fill_tensor = dct_adj(X_tensor)

    # Select the specified amount of test images
    x_gt_tensor = x_gt_tensor[:num_test_images]
    mask_tensor = mask_tensor[:num_test_images]
    X_tensor = X_tensor[:num_test_images]
    x_zero_fill_tensor = x_zero_fill_tensor[:num_test_images]

    # Define the priors
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

    # Get the model configurations
    mala_conf_dict, gibbs_config_dict, scalings, std_normal = get_model_configurations(
        dtype, device
    )

    if posterior_scales:
        scalings = {
            "normal": 3.7315000000000005,
            "laplace": 4.1093,
            "student-t": 13.8528,
            "gmm": 0.8294,
        }

    # NOTE: MALA is not used in the experiments and therefore these step-sizes are not properly tuned
    # Define the mala step sizes
    mala_step_size_dict = {
        "normal": 5.5e-5,
        "laplace": 3e-8,
        "student-t": 6e-5,
        "gmm": 4.75e-6,
    }

    # Get the data directory
    root_dir = get_data_dir()

    # Run the experiments
    print()
    print(80 * "*")
    print("Running posterior sampling experiments for DCT inpainting")
    print(80 * "*")
    for pot_name, gibbs_config in gibbs_config_dict.items():
        # Skip factors that were not specified by the configuration
        if pot_name not in factor_filter_list:
            continue

        print("\n")
        print(f"Sampling for {pot_name} factors...")
        pot, tau, scaling = (
            mala_conf_dict[pot_name],
            mala_step_size_dict[pot_name],
            scalings[pot_name],
        )

        # Get the data directory of this experiment
        if posterior_scales:
            dir = (
                    root_dir
                    / "experiments"
                    / "sampling"
                    / "posterior"
                    / "dct-inpainting-posterior-scales"
                    / pot_name
            )
        else:
            dir = (
                    root_dir
                    / "experiments"
                    / "sampling"
                    / "posterior"
                    / "dct-inpainting"
                    / pot_name
            )

        # Removing old files
        os.system(f"rm -rf {dir / '{*,.*}'}")

        # Create directory if it does not exist already
        if not os.path.exists(dir):
            os.makedirs(dir)

        # Dump the samples, the inputs and the zero fill solutions
        torch.save(x_gt_tensor, dir / "ground_truth.pth")
        torch.save(mask_tensor, dir / "masks.pth")
        torch.save(X_tensor, dir / "measurements.pth")
        torch.save(x_zero_fill_tensor, dir / "zero_fill_solutions.pth")

        for i in tqdm(range(num_test_images)):
            # Extract the current input
            mask = mask_tensor[i].clone()
            X = X_tensor[i]

            # Provide appropriate padding for broad-casting
            mask = mask[None, :, :, :]
            X = X[None, :, :, :]

            # Construct the necessary functions for MALA
            def log_p(x):
                # Extract the batch-size
                batch_size = x.shape[0]

                # Construct the energy
                R = torch.sum((pot(conv(scaling * x))).view(batch_size, -1), dim=1)
                return (
                        -0.5
                        * torch.sum((mask * x - X).view(batch_size, -1) ** 2, dim=1)
                        / (std ** 2)
                        - R
                )

            def grad_log_p(x):
                # Clone the argument
                x = torch.clone(x)
                x.requires_grad = True

                # Backprop
                loss = torch.sum(log_p(x))
                loss.backward()

                return x.grad.detach()

            # Run MALA sampling
            x_init_mala = torch.zeros(
                (num_chains, *X[0].shape), dtype=X.dtype, device=X.device
            )

            if timed_run:
                marginals_mala = torch.zeros((num_iters_mala + 1, 2, *X[0].shape), dtype=X[0].dtype, device=X[0].device)
                times_mala = torch.zeros((num_iters_mala + 1,), dtype=torch.int64, device=device)

                callback_mala = partial(
                    timed_collector, times_mala[1:], marginals_mala[1:], transform=marginal_transform
                )

            if num_iters_mala > 0:
                if timed_run:
                    times_mala[0] = time.perf_counter_ns()
                    samples_mala = mala_sampler(log_p, grad_log_p, tau, x_init_mala, num_iters_mala, False,
                                                callback_mala)
                    torch.save(marginals_mala, f"{dir}/{i:06}-marginals-mala.pth")
                    torch.save(times_mala, f"{dir}/{i:06}-times-mala.pth")
                else:
                    samples_mala = mala_sampler(log_p, grad_log_p, tau, x_init_mala, num_iters_mala, False)
                torch.save(samples_mala, f"{dir}/{i:06}-mala.pth")

            # Construct the necessary functions for Gibbs
            def K(x):
                with torch.no_grad():
                    return torch.concat((mask * dct(x), conv(scaling * x)), dim=1)

            def adj_K(y):
                with torch.no_grad():
                    y_data = y[:, :1, :, :].clone()
                    y_prior = y[:, 1:, :, :]
                    return dct_adj(mask * y_data) + adj_conv(scaling * y_prior)

            mu_map, var_map, latent_sampler = gibbs_config

            def mu_map_dct_inpainting(z):
                mean = torch.zeros(z.shape, device=device, dtype=dtype)
                mean[:, 0, :, :] = X
                mean[:, 1:, :, :] = mu_map(z[:, 1:, :, :])

                return mean

            def var_map_dct_inpainting(z):
                variances = torch.zeros(z.shape, device=device, dtype=dtype)

                # data term
                var_vals = float("inf") * torch.ones_like(variances[:, 0, :, :])
                var_vals[:, mask[0, 0]] = std ** 2
                variances[:, 0, :, :] = var_vals

                # regularizer
                variances[:, 1:, :, :] = var_map(z[:, 1:, :, :])

                return variances

            # Handle the datatype properly for the GMM factors
            if pot_name == "gmm":

                def latent_sampler_dct_inpainting(u):
                    z = torch.ones(u.shape, device=device, dtype=torch.int64)
                    z[:, 1:, :, :] = latent_sampler(u[:, 1:, :, :])

                    return z

            else:

                def latent_sampler_dct_inpainting(u):
                    z = torch.ones_like(u)
                    z[:, 1:, :, :] = latent_sampler(u[:, 1:, :, :])

                    return z

            # Run Gibbs sampling
            x_init_gibbs = torch.zeros(
                (num_chains, *X[0].shape), dtype=X.dtype, device=X.device
            )

            num_iterations_gibbs = 1 if pot_name == "normal" else num_iters_gibbs
            if timed_run:
                marginals_gibbs = torch.zeros((num_iterations_gibbs + 1, 2, *X[0].shape), dtype=X[0].dtype,
                                              device=X[0].device)
                times_gibbs = torch.zeros((num_iterations_gibbs + 1,), dtype=torch.int64, device=device)

                callback_gibbs = partial(
                    timed_collector, times_gibbs[1:], marginals_gibbs[1:], transform=marginal_transform
                )

            if num_iters_gibbs > 0:
                if timed_run:
                    times_gibbs[0] = time.perf_counter_ns()
                    samples_gibbs = gibbs_sampler(
                        K,
                        adj_K,
                        mu_map_dct_inpainting,
                        var_map_dct_inpainting,
                        latent_sampler_dct_inpainting,
                        x_init_gibbs,
                        num_iterations_gibbs,
                        False,
                        adj_K_sqr=None,
                        verbose=False,
                        callback=callback_gibbs,
                    )
                    torch.save(marginals_gibbs, f"{dir}/{i:06}-marginals-gibbs.pth")
                    torch.save(times_gibbs, f"{dir}/{i:06}-times-gibbs.pth")
                else:
                    samples_gibbs = gibbs_sampler(
                        K,
                        adj_K,
                        mu_map_dct_inpainting,
                        var_map_dct_inpainting,
                        latent_sampler_dct_inpainting,
                        x_init_gibbs,
                        num_iterations_gibbs,
                        False,
                        adj_K_sqr=None,
                        verbose=False,
                    )
                torch.save(samples_gibbs, f"{dir}/{i:06}-gibbs.pth")
        print("Done!")
    print(80 * "*")


# Main
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run posterior sampling experiments with specified parameters."
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        type=str,
        default=["denoising", "dct-inpainting"],
        help="List of inverse problems to sample from (default: ['denoising', 'dct-inpainting'])",
    )
    parser.add_argument(
        "--num_chains", type=int, default=128, help="Number of chains (default: 128)"
    )
    parser.add_argument(
        "--num_iters_mala",
        type=int,
        default=0,
        help="Number of iterations for MALA sampling (default: 0)",
    )
    parser.add_argument(
        "--num_iters_gibbs",
        type=int,
        default=3000,
        help="Number of iterations for Gibbs sampling (default: 3000)",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        type=str,
        default=["normal", "laplace", "student-t", "gmm"],
        help="List of factors to apply in the prior (default: ['normal', 'laplace', 'student-t', 'gmm'])",
    )
    parser.add_argument(
        "--num_test_images",
        type=int,
        default=256,
        help="Number of test images used for evaluation (default: 256)",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=0.1,
        help="Standard deviation of the noise (default: 0.1)",
    )
    parser.add_argument(
        "--preconditioning",
        action="store_true",
        help="Enable preconditioning (default: False)",
    )
    parser.add_argument(
        "--posterior_scales",
        action="store_true",
        help="Should posterior scales be used (default: False)",
    )
    parser.add_argument(
        "--timed_run",
        action="store_true",
        help="Should the run be timed (default: False)",
    )

    # Set experiment parameters
    args = parser.parse_args()
    problems = args.problems
    num_chains = args.num_chains
    num_iters_mala = args.num_iters_mala
    num_iters_gibbs = args.num_iters_gibbs
    factors = args.factors
    num_test_images = args.num_test_images
    std = args.std
    preconditioning = args.preconditioning
    posterior_scales = args.posterior_scales
    timed_run = args.timed_run

    # Run the posterior sampling experiments with specified parameters
    # Denoising
    if "denoising" in problems:
        run_denoising_experiments(
            num_chains,
            num_iters_mala,
            num_iters_gibbs,
            factors,
            num_test_images,
            std,
            preconditioning,
            posterior_scales,
            timed_run=timed_run
        )

    # DCT inpainting
    if "dct-inpainting" in problems:
        run_dct_inpainting_experiments(
            num_chains,
            num_iters_mala,
            num_iters_gibbs,
            factors,
            num_test_images,
            preconditioning,
            posterior_scales,
            timed_run=timed_run
        )


if __name__ == "__main__":
    # Example run command with all arguments specified:
    # python -m sampling.experiments.posteriors --problems denoising dct-inpainting --num_chains 128 --num_iters_mala
    # 0 --num_iters_gibbs 3000 --factors normal laplace student-t gmm --num_test_images 256 --std 0.1
    #
    # NOTE: Add the --preconditioning flag to enable preconditioning in the CG iterations.

    # Run the experiments
    main()
