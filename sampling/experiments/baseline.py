import os
from functools import partial

import torch
from omegaconf import OmegaConf
from torch.distributions import Exponential

from sampling.samplers import (
    mala_sampler,
    gibbs_sampler,
    potential,
    collector,
    construct_distribution,
    construct_gibbs_config,
)
from sampling.util import rng_seed, get_data_dir
import math


# Utility function to construct the linear operator for each topology
def construct_K(topology, dtype, device):
    # Basic sanity check
    assert topology in ["factor", "product", "loop", "grid"]

    # Define the linear operator in row edges then col edges ordering
    if topology == "product":
        return torch.ones((5, 1), dtype=dtype, device=device)

    if topology == "loop":
        return torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [-1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 1.0],
                [-1.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 1.0],
            ],
            dtype=dtype,
            device=device,
        )

    if topology == "grid":
        return torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 1.0],
                [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0, 1.0],
            ],
            dtype=dtype,
            device=device,
        )

    # Default for factor graphs
    return torch.ones((1, 1), dtype=dtype, device=device)


# Utility function that defines the considered models for the baseline experiment
def get_baseline_configurations(dtype, device):
    # Load the factor configurations
    factor_configs = dict(
        OmegaConf.load("sampling/experiments/configs/sampling/baseline/baseline.yaml")
    )

    # Create MALA and Gibbs configuration dictionaries
    mala_conf_dict, gibbs_config_dict = {}, {}
    for factor_name, value_dict in factor_configs.items():
        # Add to MALA configurations
        distribution = construct_distribution(factor_name, value_dict, dtype, device)
        mala_conf_dict[factor_name] = partial(potential, distribution)

        # Add to Gibbs configurations
        gibbs_config_dict[factor_name] = construct_gibbs_config(
            factor_name, value_dict, dtype, device
        )

    return mala_conf_dict, gibbs_config_dict


# Utility function that defines the considered models for the baseline tail experiment
def get_tail_configurations(dtype, device):
    # Load degree of freedom list for this experiment
    df_list = dict(
        OmegaConf.load("sampling/experiments/configs/sampling/baseline/tails.yaml")
    )["df_list"]

    # Create MALA and Gibbs configuration dictionaries
    mala_conf_dict, gibbs_config_dict = {}, {}
    for df in df_list:
        # Add to MALA configurations
        distribution = construct_distribution("student-t", {"df": df}, dtype, device)
        mala_conf_dict[df] = partial(potential, distribution)

        # Add to Gibbs configurations
        gibbs_config_dict[df] = construct_gibbs_config(
            "student-t", {"df": df}, dtype, device
        )

    return mala_conf_dict, gibbs_config_dict


# Utility function that defines the considered models for the baseline approximation experiment
def get_approximation_configurations(type, dtype, device):
    if type == "uniform":
        path = (
            "sampling/experiments/configs/sampling/baseline/approximation_uniform.yaml"
        )
    else:
        path = "sampling/experiments/configs/sampling/baseline/approximation_gsm.yaml"

    # Load the number of GMM components used for approximation
    num_components_list = dict(OmegaConf.load(path))["num_components_list"]

    # Create MALA and Gibbs configuration dictionaries
    mala_conf_dict, gibbs_config_dict = {}, {}
    for num_components in num_components_list:
        # Construct the potential
        hr = 0.5
        if type == "uniform":
            # Set up means and variances
            means = torch.linspace(-hr, hr, num_components, dtype=dtype, device=device)
            vars = (2 * hr / (num_components - 1)) ** 2 * torch.ones_like(means)

            # Compute weights to match a Laplace distribution
            weights = torch.exp(-torch.abs(means))
        else:
            # Discretize the latent distribution through its inverse CDF to compute the variance of the GMM components
            latent_dist = Exponential(torch.tensor(0.5))
            ptile = 0.5e-7
            probs = torch.linspace(ptile, 1 - ptile, num_components + 1, dtype=dtype)
            z = latent_dist.icdf(probs)
            vars = (z[1:] + z[:-1]) / 2.0
            means = torch.zeros_like(vars)

            # Compute weights to match a Laplace distribution
            f = torch.exp(latent_dist.log_prob(z))
            weights = 0.5 * (f[1:] + f[:-1]) * (z[1:] - z[:-1])

        # Add to MALA configurations
        value_dict = {
            "weights": [weights.tolist()],
            "means": [means.tolist()],
            "vars": [vars.tolist()],
        }
        distribution = construct_distribution("gmm", value_dict, dtype, device)
        mala_conf_dict[num_components] = partial(potential, distribution)

        # Add to Gibbs configurations
        gibbs_config_dict[num_components] = construct_gibbs_config(
            "gmm", value_dict, dtype, device
        )

    return mala_conf_dict, gibbs_config_dict


def get_initialization_configurations(dtype, device):
    # Load degree of freedom list for this experiment
    config = dict(
        OmegaConf.load(
            "sampling/experiments/configs/sampling/baseline/initialization.yaml"
        )
    )
    df, norm_list = config["student-t"]["df"], config["norm_list"]

    # Create MALA and Gibbs configuration dictionaries
    mala_conf_dict, gibbs_config_dict = {}, {}
    for norm in norm_list:
        # Add to MALA configurations
        distribution = construct_distribution("student-t", {"df": df}, dtype, device)
        mala_conf_dict[norm] = partial(potential, distribution)

        # Add to Gibbs configurations
        gibbs_config_dict[norm] = construct_gibbs_config(
            "student-t", {"df": df}, dtype, device
        )

    return mala_conf_dict, gibbs_config_dict


# Experiment runner for the baseline experiment
def run_baseline_experiment(num_chains, num_iters, dtype, device):
    # Basic sanity checks
    assert num_chains > 0
    assert num_iters > 0

    # Run the experiments
    topology_list = ["factor", "product", "loop", "grid"]

    # Get model configurations
    mala_conf_dict, gibbs_config_dict = get_baseline_configurations(dtype, device)

    # Define the mala step sizes
    factor_mala_step_size_dict = {
        "normal": 2.75e-1,
        "laplace": 2e-1,
        "student-t": 2.25,
        "gmm": 1.35e-4,
    }
    product_mala_step_size_dict = {
        "normal": 5.5e-2,
        "laplace": 7.75e-3,
        "student-t": 3.25e-1,
        "gmm": 1e-5,
    }
    loop_mala_step_size_dict = {
        "normal": 4.5e-2,
        "laplace": 1.75e-2,
        "student-t": 3.25e-1,
        "gmm": 6e-5,
    }
    grid_mala_step_size_dict = {
        "normal": 3.25e-2,
        "laplace": 1.15e-2,
        "student-t": 2.25e-1,
        "gmm": 3.25e-5,
    }

    mala_step_size_dict = {
        "factor": factor_mala_step_size_dict,
        "product": product_mala_step_size_dict,
        "loop": loop_mala_step_size_dict,
        "grid": grid_mala_step_size_dict,
    }

    # Get the data directory
    root_dir = get_data_dir()

    print()
    print(80 * "*")
    print("Running baseline sampling experiments")
    print(80 * "*")
    for topology in topology_list:
        # Get the linear operator and it's adjoint for the current topology
        K_matrix = construct_K(topology, dtype, device)

        for pot_name, gibbs_config in gibbs_config_dict.items():
            print()
            print(f"Sampling for {topology} graph and {pot_name} factors...")

            # Define the initial value
            x_init = torch.zeros(
                (num_chains, K_matrix.shape[1]), dtype=dtype, device=device
            )

            # Construct callbacks to collect the samples
            samples_mala = torch.zeros(
                (num_iters + 1, *x_init.shape), dtype=dtype, device=device
            )
            samples_gibbs = torch.zeros(
                (num_iters + 1, *x_init.shape), dtype=dtype, device=device
            )

            callback_mala = partial(collector, samples_mala[1:])
            callback_gibbs = partial(collector, samples_gibbs[1:])

            # Construct the necessary functions for MALA
            pot, tau = mala_conf_dict[pot_name], mala_step_size_dict[topology][pot_name]

            def energy(x):
                batch_size = x.shape[0]

                # Compute the model energy
                u = x @ K_matrix.t()
                model_energy = torch.sum(
                    pot(u[:, :, None, None]).view(batch_size, -1), dim=1
                )

                return model_energy

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

            # Run MALA sampler
            rng_seed()
            mala_sampler(log_p, grad_log_p, tau, x_init, num_iters, True, callback_mala)

            # Construct the necessary functions for Gibbs
            mu_map, var_map, latent_sampler = gibbs_config

            def K(x):
                return x @ K_matrix.t()

            def adj_K(y):
                return y @ K_matrix

            # Get the data directory of this experiment
            dir = (
                root_dir
                / "experiments"
                / "sampling"
                / "baseline"
                / "baseline"
                / topology
                / pot_name
            )

            # Removing old files
            os.system(f"rm -rf {dir / '{*,.*}'}")

            # Dump MALA sampling results to disk
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save(samples_mala, dir / "samples_mala.pth")

            # Run Gibbs sampler
            rng_seed()
            gibbs_sampler(
                K,
                adj_K,
                mu_map,
                var_map,
                latent_sampler,
                x_init,
                num_iters,
                False,
                callback_gibbs,
            )

            # Dump Gibbs sampling results to disk
            torch.save(samples_gibbs, dir / "samples_gibbs.pth")

            print("Done!")
    print(80 * "*")


# Experiment runner for the tail experiment
def run_tail_experiment(num_chains, num_iters, dtype, device):
    # Basic sanity checks
    assert num_chains > 0
    assert num_iters > 0

    # Run the experiments
    topology_list = ["factor", "product", "loop", "grid"]

    # Get model configurations
    mala_conf_dict, gibbs_config_dict = get_tail_configurations(dtype, device)

    # Define the mala step sizes
    factor_mala_step_size_dict = {2: 2.75, 3: 2.5, 4: 2.25, 5: 2.125, 6: 2.0}
    product_mala_step_size_dict = {
        2: 2.75e-1,
        3: 2.85e-1,
        4: 3e-1,
        5: 3.125e-1,
        6: 3.25e-1,
    }
    loop_mala_step_size_dict = {2: 3.5e-1, 3: 3.4e-1, 4: 3.3e-1, 5: 3.25e-1, 6: 3.2e-1}
    grid_mala_step_size_dict = {
        2: 2.375e-1,
        3: 2.35e-1,
        4: 2.325e-1,
        5: 2.3e-1,
        6: 2.275e-1,
    }

    mala_step_size_dict = {
        "factor": factor_mala_step_size_dict,
        "product": product_mala_step_size_dict,
        "loop": loop_mala_step_size_dict,
        "grid": grid_mala_step_size_dict,
    }

    # Get the data directory
    root_dir = get_data_dir()

    print()
    print(80 * "*")
    print("Running baseline tail sampling experiments")
    print(80 * "*")
    for topology in topology_list:
        # Get the linear operator and it's adjoint for the current topology
        K_matrix = construct_K(topology, dtype, device)

        for df, gibbs_config in gibbs_config_dict.items():
            print()
            print(
                f"Sampling for {topology} graph and student-t factors with df = {df}..."
            )

            # Define the initial value
            x_init = torch.zeros(
                (num_chains, K_matrix.shape[1]), dtype=dtype, device=device
            )

            # Construct callbacks to collect the samples
            samples_mala = torch.zeros(
                (num_iters + 1, *x_init.shape), dtype=dtype, device=device
            )
            samples_gibbs = torch.zeros(
                (num_iters + 1, *x_init.shape), dtype=dtype, device=device
            )

            callback_mala = partial(collector, samples_mala[1:])
            callback_gibbs = partial(collector, samples_gibbs[1:])

            # Construct the necessary functions for MALA
            pot, tau = mala_conf_dict[df], mala_step_size_dict[topology][df]

            def energy(x):
                batch_size = x.shape[0]

                # Compute the model energy
                u = x @ K_matrix.t()
                model_energy = torch.sum(
                    pot(u[:, :, None, None]).view(batch_size, -1), dim=1
                )

                return model_energy

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

            # Run MALA sampler
            rng_seed()
            mala_sampler(log_p, grad_log_p, tau, x_init, num_iters, True, callback_mala)

            # Construct the necessary functions for Gibbs
            mu_map, var_map, latent_sampler = gibbs_config

            def K(x):
                return x @ K_matrix.t()

            def adj_K(y):
                return y @ K_matrix

            # Get the data directory of this experiment
            dir = (
                root_dir
                / "experiments"
                / "sampling"
                / "baseline"
                / "tails"
                / topology
                / f"{df}"
            )

            # Removing old files
            os.system(f"rm -rf {dir / '{*,.*}'}")

            # Dump MALA sampling results to disk
            torch.save(samples_mala, dir / "samples_mala.pth")

            # Run Gibbs sampler
            rng_seed()
            gibbs_sampler(
                K,
                adj_K,
                mu_map,
                var_map,
                latent_sampler,
                x_init,
                num_iters,
                False,
                callback_gibbs,
            )

            # Dump Gibbs sampling results to disk
            torch.save(samples_gibbs, dir / "samples_gibbs.pth")

            print("Done!")
    print(80 * "*")


# Experiment runner for the approximation experiments
def run_approximation_experiment(type, num_chains, num_iters, dtype, device):
    # Basic sanity checks
    assert type in ["uniform", "gsm"]
    assert num_chains > 0
    assert num_iters > 0

    # Run the experiments
    topology_list = ["factor", "product", "loop", "grid"]

    # Get model configurations
    mala_conf_dict, gibbs_config_dict = get_approximation_configurations(
        type, dtype, device
    )

    # Define the mala step sizes
    if type == "uniform":
        factor_mala_step_size_dict = {512: 0.15, 1024: 0.15, 1536: 0.15, 2048: 0.15}
        product_mala_step_size_dict = {
            512: 0.0675,
            1024: 0.0675,
            1536: 0.0675,
            2048: 0.0675,
        }
        loop_mala_step_size_dict = {512: 0.008, 1024: 0.008, 1536: 0.008, 2048: 0.008}
        grid_mala_step_size_dict = {
            512: 0.00375,
            1024: 0.00375,
            1536: 0.00375,
            2048: 0.00375,
        }
    else:
        factor_mala_step_size_dict = {256: 2.25, 512: 2.25, 1024: 2.25, 2048: 2.25}
        product_mala_step_size_dict = {256: 0.085, 512: 0.085, 1024: 0.085, 2048: 0.085}
        loop_mala_step_size_dict = {256: 0.2, 512: 0.2, 1024: 0.2, 2048: 0.2}
        grid_mala_step_size_dict = {
            256: 0.1325,
            512: 0.1325,
            1024: 0.1325,
            2048: 0.1325,
        }

    mala_step_size_dict = {
        "factor": factor_mala_step_size_dict,
        "product": product_mala_step_size_dict,
        "loop": loop_mala_step_size_dict,
        "grid": grid_mala_step_size_dict,
    }

    # Get the data directory
    root_dir = get_data_dir()

    print()
    print(80 * "*")
    print(f"Running baseline {type} approximation sampling experiments")
    print(80 * "*")
    for topology in topology_list:
        # Get the linear operator and it's adjoint for the current topology
        K_matrix = construct_K(topology, dtype, device)

        for num_components, gibbs_config in gibbs_config_dict.items():
            print()
            print(
                f"Sampling for {topology} graph and GMM factors with {num_components} components..."
            )

            # Define the initial value
            x_init = torch.zeros(
                (num_chains, K_matrix.shape[1]), dtype=dtype, device=device
            )

            # Construct callbacks to collect the samples
            samples_mala = torch.zeros(
                (num_iters + 1, *x_init.shape), dtype=dtype, device=device
            )
            samples_gibbs = torch.zeros(
                (num_iters + 1, *x_init.shape), dtype=dtype, device=device
            )

            callback_mala = partial(collector, samples_mala[1:])
            callback_gibbs = partial(collector, samples_gibbs[1:])

            # Construct the necessary functions for MALA
            pot, tau = (
                mala_conf_dict[num_components],
                mala_step_size_dict[topology][num_components],
            )

            def energy(x):
                batch_size = x.shape[0]

                # Compute the model energy
                u = x @ K_matrix.t()
                model_energy = torch.sum(
                    pot(u[:, :, None, None]).view(batch_size, -1), dim=1
                )

                return model_energy

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

            # Run MALA sampler
            rng_seed()
            mala_sampler(log_p, grad_log_p, tau, x_init, num_iters, True, callback_mala)

            # Construct the necessary functions for Gibbs
            mu_map, var_map, latent_sampler = gibbs_config

            def K(x):
                return x @ K_matrix.t()

            def adj_K(y):
                return y @ K_matrix

            # Get the data directory of this experiment
            dir = (
                root_dir
                / "experiments"
                / "sampling"
                / "baseline"
                / "approximation"
                / f"{type}"
                / topology
                / f"{num_components}"
            )

            # Removing old files
            os.system(f"rm -rf {dir / '{*,.*}'}")

            # Dump MALA sampling results to disk
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save(samples_mala, dir / "samples_mala.pth")

            # Run Gibbs sampler
            rng_seed()
            gibbs_sampler(
                K,
                adj_K,
                mu_map,
                var_map,
                latent_sampler,
                x_init,
                num_iters,
                False,
                callback_gibbs,
            )

            # Dump Gibbs sampling results to disk
            torch.save(samples_gibbs, dir / "samples_gibbs.pth")

            print("Done!")
    print(80 * "*")


# Experiment runner for the baseline initialization experiment
def run_initialization_experiment(num_chains, num_iters, dtype, device):
    # Basic sanity checks
    assert num_chains > 0
    assert num_iters > 0

    # Run the experiments
    topology_list = ["factor", "product", "loop", "grid"]

    # Get model configurations
    mala_conf_dict, gibbs_config_dict = get_initialization_configurations(dtype, device)

    # Define the mala step sizes
    factor_mala_step_size_dict = {1: 2.25, 5: 2.25, 10: 2.25, 15: 2.25}
    product_mala_step_size_dict = {1: 3.25e-1, 5: 3.25e-1, 10: 3.25e-1, 15: 3.25e-1}
    loop_mala_step_size_dict = {1: 3.25e-1, 5: 3.25e-1, 10: 3.25e-1, 15: 3.25e-1}
    grid_mala_step_size_dict = {1: 2.25e-1, 5: 2.25e-1, 10: 2.25e-1, 15: 2.25e-1}

    mala_step_size_dict = {
        "factor": factor_mala_step_size_dict,
        "product": product_mala_step_size_dict,
        "loop": loop_mala_step_size_dict,
        "grid": grid_mala_step_size_dict,
    }

    # Get the data directory
    root_dir = get_data_dir()

    print()
    print(80 * "*")
    print("Running baseline sampling initialization experiments")
    print(80 * "*")
    for topology in topology_list:
        # Get the linear operator and it's adjoint for the current topology
        K_matrix = construct_K(topology, dtype, device)

        for norm, gibbs_config in gibbs_config_dict.items():
            print()
            print()
            print(
                f"Sampling for {topology} graph and student-t factors with init norm = {norm}..."
            )

            # Define the initial value
            val = norm / math.sqrt(K_matrix.shape[1])
            x_init = (
                torch.zeros((num_chains, K_matrix.shape[1]), dtype=dtype, device=device)
                + val
            )

            # Construct callbacks to collect the samples
            samples_mala = torch.zeros(
                (num_iters + 1, *x_init.shape), dtype=dtype, device=device
            )
            samples_gibbs = torch.zeros(
                (num_iters + 1, *x_init.shape), dtype=dtype, device=device
            )

            callback_mala = partial(collector, samples_mala[1:])
            callback_gibbs = partial(collector, samples_gibbs[1:])

            # Construct the necessary functions for MALA
            pot, tau = mala_conf_dict[norm], mala_step_size_dict[topology][norm]

            def energy(x):
                batch_size = x.shape[0]

                # Compute the model energy
                u = x @ K_matrix.t()
                model_energy = torch.sum(
                    pot(u[:, :, None, None]).view(batch_size, -1), dim=1
                )

                return model_energy

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

            # Run MALA sampler
            rng_seed()
            mala_sampler(log_p, grad_log_p, tau, x_init, num_iters, True, callback_mala)

            # Construct the necessary functions for Gibbs
            mu_map, var_map, latent_sampler = gibbs_config

            def K(x):
                return x @ K_matrix.t()

            def adj_K(y):
                return y @ K_matrix

            # Get the data directory of this experiment
            dir = (
                root_dir
                / "experiments"
                / "sampling"
                / "baseline"
                / "initialization"
                / topology
                / f"{norm}"
            )

            # Removing old files
            os.system(f"rm -rf {dir / '{*,.*}'}")

            # Creating folder in case it did not exist
            if not os.path.exists(dir):
                os.makedirs(dir)

            # Dump MALA sampling results to disk
            torch.save(samples_mala, dir / "samples_mala.pth")

            # Run Gibbs sampler
            rng_seed()
            gibbs_sampler(
                K,
                adj_K,
                mu_map,
                var_map,
                latent_sampler,
                x_init,
                num_iters,
                False,
                callback_gibbs,
            )

            # Dump Gibbs sampling results to disk
            torch.save(samples_gibbs, dir / "samples_gibbs.pth")

            print("Done!")
    print(80 * "*")


# Experiment runner function
def run_experiments():
    # Define the dtype and device
    dtype = torch.float32
    device = "cuda"

    # Define the number of chains and samples for the samplers
    num_chains, num_iters = 10_000, 15_000

    # Run baseline experiment
    run_baseline_experiment(num_chains, num_iters, dtype, device)

    # Run tail experiment
    run_tail_experiment(num_chains, num_iters, dtype, device)

    # Run uniform approximation experiment
    run_approximation_experiment("uniform", num_chains, num_iters, dtype, device)

    # Run GSM approximation experiment
    run_approximation_experiment("gsm", num_chains, num_iters, dtype, device)

    # Run initialization experiment
    run_initialization_experiment(num_chains, num_iters, dtype, device)


if __name__ == "__main__":
    # Run the experiments
    run_experiments()
