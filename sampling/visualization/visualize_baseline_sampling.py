import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from statsmodels.tsa.stattools import acf
from torch.distributions import Exponential

from sampling.experiments.baseline import construct_K
from sampling.samplers import construct_distribution
from sampling.util import (
    get_data_dir,
    init_matplotlib,
    wasserstein_1_distance,
    zoh,
    is_finite,
    capitalize,
    extend_np_array,
)


########################################################################################################################
# Utility functions
########################################################################################################################
# Utility function that computes the marginals from the samples
def compute_sample_marginals(topology, samples):
    # Basic sanity check
    assert topology in ["factor", "product", "loop", "grid"]

    if topology in ["factor", "product"]:
        return samples[:, :, 0]

    if topology == "loop":
        # Compute the marginals
        K_matrix = construct_K(topology, torch.float64, samples.device)
        shape = samples.shape
        marginals = (samples.reshape(-1, 4).to(torch.float64) @ K_matrix.t()).reshape(
            (shape[0], shape[1], 5)
        )

        # Ignore the tie-breaking term
        marginals = marginals[:, :, 1:]

        # The last four dimensions describe marginals that come from the same distribution, so we collect them together
        return torch.cat(
            (
                marginals[:, :, 0],
                marginals[:, :, 1],
                marginals[:, :, 2],
                marginals[:, :, 3],
            ),
            dim=1,
        )

    # Grid as default option
    # Compute the marginals
    K_matrix = construct_K(topology, torch.float64, samples.device)
    shape = samples.shape
    marginals = (samples.reshape(-1, 6).to(torch.float64) @ K_matrix.t()).reshape(
        (shape[0], shape[1], 8)
    )

    # Ignore the tie-breaking term
    marginals = marginals[:, :, 1:]

    # Extract the inner marginals
    inner_marginals = marginals[:, :, 5]

    # The remaining last dimensions describe marginals that come from the same distribution, so we collect them together
    outer_marginals = torch.cat(
        (
            marginals[:, :, 0],
            marginals[:, :, 1],
            marginals[:, :, 2],
            marginals[:, :, 3],
            marginals[:, :, 4],
            marginals[:, :, 6],
        ),
        dim=1,
    )

    return inner_marginals, outer_marginals


########################################################################################################################
# Ground-truth computations
########################################################################################################################
# Utility function that converts a torch distribution factor to a numpy factor
def factor_to_numpy(factor):
    return lambda arg: (torch.exp(factor.log_prob(torch.from_numpy(arg)))).numpy()


# Ground-truth marginal computation for product graph
def main_baseline_product(fpdf, xhr, n):
    # Basic sanity checks
    assert xhr > 0.0
    assert n > 0

    # Define the grid for the numerical ground-truth evaluation
    x = np.linspace(-xhr, xhr, n)

    # Compute the ground-truth marginal distribution
    f_product = fpdf(x) * fpdf(x) * fpdf(x) * fpdf(x) * fpdf(x)
    dx = np.diff(x)[0]
    f_product /= np.sum(f_product * dx)

    return x, f_product


# Ground-truth marginal computation for loop graph
def main_baseline_loop(fpdf, xhr, n):
    # Basic sanity checks
    assert xhr > 0.0
    assert n > 0

    # Define the grid for the numerical ground-truth evaluation
    x = np.linspace(-xhr, xhr, n)

    # Compute the ground-truth marginal distribution
    f_loop = np.convolve(fpdf(x), fpdf(x))
    x = np.linspace(-2 * xhr, 2 * xhr, f_loop.size)

    f_loop = np.convolve(f_loop, fpdf(x))
    x = np.linspace(-4 * xhr, 4 * xhr, f_loop.size)

    f_loop *= fpdf(x)
    dx = np.diff(x)[0]
    f_loop /= np.sum(f_loop * dx)

    return x, f_loop


# Ground-truth marginal computation for 2x3 grid graph
def main_baseline_2x3_grid(fpdf, xhr, n):
    # Basic sanity checks
    assert xhr > 0.0
    assert n > 0

    # Define the grid for the numerical ground-truth evaluation
    x_inner = np.linspace(-xhr, xhr, n)
    x_outer = np.linspace(-xhr, xhr, n)

    # Compute the inner ground-truth marginal distribution
    f_grid_inner = np.convolve(fpdf(x_inner), fpdf(x_inner))
    x_inner = np.linspace(-2 * xhr, 2 * xhr, f_grid_inner.size)

    f_grid_inner = np.convolve(f_grid_inner, fpdf(x_inner))
    x_inner = np.linspace(-4 * xhr, 4 * xhr, f_grid_inner.size)

    f_grid_inner *= f_grid_inner
    f_grid_inner *= fpdf(x_inner)
    dx_inner = np.diff(x_inner)[0]
    f_grid_inner /= np.sum(f_grid_inner * dx_inner)

    # Compute the outer ground-truth marginal distribution
    f_grid_outer = np.convolve(fpdf(x_outer), fpdf(x_outer))
    x_outer = np.linspace(-2 * xhr, 2 * xhr, f_grid_outer.size)

    f_grid_outer = np.convolve(f_grid_outer, fpdf(x_outer))
    x_outer = np.linspace(-4 * xhr, 4 * xhr, f_grid_outer.size)

    f_grid_outer *= fpdf(x_outer)

    f_grid_outer = np.convolve(f_grid_outer, fpdf(x_outer))
    x_outer = np.linspace(-8 * xhr, 8 * xhr, f_grid_outer.size)

    f_grid_outer = np.convolve(f_grid_outer, fpdf(x_outer))
    x_outer = np.linspace(-16 * xhr, 16 * xhr, f_grid_outer.size)

    f_grid_outer *= fpdf(x_outer)
    dx_outer = np.diff(x_outer)[0]
    f_grid_outer /= np.sum(f_grid_outer * dx_outer)

    return x_inner, f_grid_inner, x_outer, f_grid_outer


########################################################################################################################
# Plotting of sampling results
########################################################################################################################
# Computes and plots the ground-truth marginals in all baseline experiments
def plot_ground_truth_marginals(type, n=10_000):
    # Basic sanity check
    assert type in [
        "baseline",
        "tails",
        "initialization",
        "approximation-uniform",
        "approximation-gsm",
    ]

    # Read the model configurations
    dtype, device = torch.float64, "cpu"

    # Select between tail and baseline ground-truth computations
    if type == "tails":
        factor_configs = dict(
            OmegaConf.load("sampling/experiments/configs/sampling/baseline/tails.yaml")
        )
        df_list, hr_list = factor_configs["df_list"], factor_configs["hr_list"]

        # Basic sanity check
        assert len(df_list) == len(hr_list)

        # Load the list of factors into a dictionary
        factor_dict = {}
        for i, (df, hr) in enumerate(zip(df_list, hr_list)):
            # Basic sanity checks
            assert df > 0.0
            assert hr > 0.0

            # Add the factor to the dictionary
            factor_dict[df] = {
                "factor": construct_distribution(
                    "student-t", {"df": df}, dtype, device
                ),
                "hr": hr,
            }
    elif type == "baseline":
        factor_configs = dict(
            OmegaConf.load(
                "sampling/experiments/configs/sampling/baseline/baseline.yaml"
            )
        )

        # Load the list of factors into a dictionary
        factor_dict = {}
        for key, value_dict in factor_configs.items():
            # Basic sanity check
            assert value_dict["hr"] > 0.0

            # Add the factor to the dictionary
            factor_dict[key] = {
                "factor": construct_distribution(key, value_dict, dtype, device),
                "hr": value_dict["hr"],
            }
    elif type == "initialization":
        config = dict(
            OmegaConf.load(
                "sampling/experiments/configs/sampling/baseline/initialization.yaml"
            )
        )
        df, hr, norm_list = (
            config["student-t"]["df"],
            config["student-t"]["hr"],
            config["norm_list"],
        )
        # Basic sanity checks
        assert df > 0.0
        assert hr > 0.0

        # Load the list of factors into a dictionary
        factor_dict = {}
        for norm in norm_list:
            # Basic sanity checks
            assert df > 0.0
            assert hr > 0.0

            # Add the factor to the dictionary
            factor_dict[norm] = {
                "factor": construct_distribution(
                    "student-t", {"df": df}, dtype, device
                ),
                "hr": hr,
            }
    else:
        if type == "approximation-uniform":
            path = "sampling/experiments/configs/sampling/baseline/approximation_uniform.yaml"
        else:
            path = (
                "sampling/experiments/configs/sampling/baseline/approximation_gsm.yaml"
            )

        # Load the number of GMM components used for approximation
        num_components_list = dict(OmegaConf.load(path))["num_components_list"]
        hr_list = dict(OmegaConf.load(path))["hr_list"]

        # Basic sanity check
        assert len(num_components_list) == len(hr_list)

        # Load the list of factors into a dictionary
        factor_dict = {}
        for num_components, hr in zip(num_components_list, hr_list):
            # Basic sanity checks
            assert num_components > 0
            assert hr > 0.0

            # Construct the potential
            if type == "approximation-uniform":
                # Set up means and variances
                means = torch.linspace(
                    -0.5, 0.5, num_components, dtype=dtype, device=device
                )
                vars = (1.0 / (num_components - 1)) ** 2 * torch.ones_like(means)

                # Compute weights to match a Laplace distribution
                weights = torch.exp(-torch.abs(means))
            else:
                # Discretize the latent distribution through its inverse CDF to compute the variance of the GMM components
                latent_dist = Exponential(torch.tensor(0.5))
                ptile = 0.5e-7
                probs = torch.linspace(
                    ptile, 1 - ptile, num_components + 1, dtype=dtype
                )
                z = latent_dist.icdf(probs)
                vars = (z[1:] + z[:-1]) / 2.0
                means = torch.zeros_like(vars)

                # Compute weights to match a Laplace distribution
                f = torch.exp(latent_dist.log_prob(z))
                weights = 0.5 * (f[1:] + f[:-1]) * (z[1:] - z[:-1])

            # Add the factor to the dictionary
            value_dict = {
                "weights": [weights.tolist()],
                "means": [means.tolist()],
                "vars": [vars.tolist()],
            }
            factor_dict[num_components] = {
                "factor": construct_distribution("gmm", value_dict, dtype, device),
                "hr": hr,
            }

    # Define the topologies
    topology_run_dict = {
        "factor": None,
        "product": main_baseline_product,
        "loop": main_baseline_loop,
        "grid": main_baseline_2x3_grid,
    }

    # Get the data directory
    root_dir = get_data_dir()

    # Get the output directory
    if type == "tails":
        dir = root_dir / "experiments" / "sampling" / "baseline" / "tails"
    elif type == "baseline":
        dir = root_dir / "experiments" / "sampling" / "baseline" / "baseline"
    elif type == "initialization":
        dir = root_dir / "experiments" / "sampling" / "baseline" / "initialization"
    elif type == "approximation-uniform":
        dir = (
            root_dir
            / "experiments"
            / "sampling"
            / "baseline"
            / "approximation"
            / "uniform"
        )
    else:
        dir = (
            root_dir / "experiments" / "sampling" / "baseline" / "approximation" / "gsm"
        )

    # Compute the ground-truth
    print()
    print(80 * "*")
    if type == "tails":
        print(
            "Running ground-truth marginal computations for baseline tail experiments"
        )
    elif type == "baseline":
        print("Running ground-truth marginal computations for baseline experiments")
    elif type == "initialization":
        print(
            "Running ground-truth marginal computations for baseline initialization experiments"
        )
    elif type == "approximation-uniform":
        print(
            "Running ground-truth marginal computations for baseline uniform approximation experiments"
        )
    else:
        print(
            "Running ground-truth marginal computations for baseline gsm approximation experiments"
        )
    print(80 * "*")

    # Run the experiments
    rows, cols = len(topology_run_dict), len(factor_dict)
    fig1 = plt.figure(figsize=(4.5 * cols, 4.5 * rows))
    fig2 = plt.figure(figsize=(4.5 * cols, 4.5 * rows))
    for i, (topology, marginals) in enumerate(topology_run_dict.items()):
        for j, (key, factor) in enumerate(factor_dict.items()):
            print()
            if type == "tails":
                print(
                    f"Computing for {topology} graph and student-t factors with df = {key}..."
                )
            elif type == "baseline":
                print(f"Computing for {topology} graph and {key} factors...")
            elif type == "initialization":
                print(
                    f"Computing for {topology} graph and student-t factors with init norm = {key}..."
                )
            else:
                print(
                    f"Computing for {topology} graph and GMM factors with {key} components..."
                )

            idx = i * cols + j + 1
            plt.figure(fig1.number)
            plt.subplot(rows, cols, idx)

            if i == 0:
                if type == "tails":
                    plt.title(rf"$\nu = {key}$")
                elif type == "baseline":
                    plt.title(capitalize(key))
                elif type == "initialization":
                    plt.title(rf"$c = {key}$")
                else:
                    plt.title(f"${key}$")

            if j == 0:
                plt.ylabel(capitalize(topology))

            plt.figure(fig2.number)
            plt.subplot(rows, cols, idx)

            if i == 0:
                if type == "tails":
                    plt.title(rf"$\nu = {key}$")
                elif type == "baseline":
                    plt.title(key)
                elif type == "initialization":
                    plt.title(rf"$\|x_0\| = {key}$")
                else:
                    plt.title(f"${key}$")

            if j == 0:
                plt.ylabel(topology)

            # Get the dir of the current experiment
            experiment_dir = dir / topology / f"{key}"

            # Compute the marginals for the current topology and factor and plot them
            xhr = factor_dict[key]["hr"]

            # Load the MALA and Gibbs samples
            samples_mala = torch.load(
                experiment_dir / "samples_mala.pth", weights_only=True
            )[-1:]
            samples_gibbs = torch.load(
                experiment_dir / "samples_gibbs.pth", weights_only=True
            )[-1:]

            # Basic sanity checks
            assert is_finite(samples_mala)
            assert is_finite(samples_gibbs)

            # Load the samples
            if topology != "grid":
                # Compute the marginals of the samples
                marginals_mala = compute_sample_marginals(topology, samples_mala)[0]
                marginals_gibbs = compute_sample_marginals(topology, samples_gibbs)[0]

                # Sort the sample marginals
                marginals_mala = torch.sort(marginals_mala)[0].cpu()
                marginals_gibbs = torch.sort(marginals_gibbs)[0].cpu()

                # Construct the empirical CDF
                emp_cdf = torch.cumsum(
                    torch.ones_like(marginals_mala) / marginals_mala.numel(), dim=0
                )
            else:
                # Compute the marginals of the samples
                inner_marginals_mala, outer_marginals_mala = compute_sample_marginals(
                    topology, samples_mala
                )
                inner_marginals_gibbs, outer_marginals_gibbs = compute_sample_marginals(
                    topology, samples_gibbs
                )
                inner_marginals_mala = inner_marginals_mala[0]
                outer_marginals_mala = outer_marginals_mala[0]
                inner_marginals_gibbs = inner_marginals_gibbs[0]
                outer_marginals_gibbs = outer_marginals_gibbs[0]

                # Sort the sample marginals
                inner_marginals_mala = torch.sort(inner_marginals_mala)[0].cpu()
                outer_marginals_mala = torch.sort(outer_marginals_mala)[0].cpu()
                inner_marginals_gibbs = torch.sort(inner_marginals_gibbs)[0].cpu()
                outer_marginals_gibbs = torch.sort(outer_marginals_gibbs)[0].cpu()

                # Construct the empirical CDFs
                emp_cdf_inner = torch.cumsum(
                    torch.ones_like(inner_marginals_mala)
                    / inner_marginals_mala.numel(),
                    dim=0,
                )
                emp_cdf_outer = torch.cumsum(
                    torch.ones_like(outer_marginals_mala)
                    / outer_marginals_mala.numel(),
                    dim=0,
                )

            ptile = 0.001
            if topology == "factor":
                x = np.linspace(-xhr, xhr, n)
                f = factor_to_numpy(factor["factor"])(x)

                plt.figure(fig1.number)
                if type == "approximation-uniform":
                    idx = f > 0.0
                else:
                    if type == "baseline" and key == "gmm":
                        idx = np.where((x >= -0.5) & (x <= 0.5))
                    else:
                        idx = np.where((x >= -xhr) & (x <= xhr))

                val = -np.log(f[idx])
                plt.plot(x[idx], val)

                pd.DataFrame({"x": x[idx], "y": val}).to_csv(
                    dir / f"ground-truth_{i + 1}_{j + 1}.csv", index=False
                )

                vlb, vub = val.min().item(), val.max().item()
                plt.ylim([vlb - 0.025 * (vub - vlb), vub])

                plt.figure(fig2.number)
                x = np.linspace(-xhr, xhr, n)
                f = factor_to_numpy(factor["factor"])(x)
                cdf = np.cumsum(f)
                cdf /= cdf[-1]
                lb_idx, ub_idx = np.argmax(cdf >= ptile), np.argmin(cdf <= 1 - ptile)
                xlb, xub = x[lb_idx], x[ub_idx]

                if type == "approximation-uniform":
                    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                    zero = torch.tensor(
                        [0.0], dtype=emp_cdf.dtype, device=emp_cdf.device
                    )
                    one = torch.tensor(
                        [1.0], dtype=emp_cdf.dtype, device=emp_cdf.device
                    )
                    emp_cdf_cat = torch.cat((zero, emp_cdf, one))
                    mala_cat = torch.cat(
                        (-one * 10 * hr, marginals_mala, one * 10 * hr)
                    )
                    gibbs_cat = torch.cat(
                        (-one * 10 * hr, marginals_gibbs, one * 10 * hr)
                    )
                    plt.plot(mala_cat, emp_cdf_cat, color=cycle[0])
                    plt.plot(gibbs_cat, emp_cdf_cat, "--", color=cycle[0])
                    pd.DataFrame(
                        {
                            "x": extend_np_array(mala_cat),
                            "y": extend_np_array(emp_cdf_cat),
                        }
                    ).to_csv(
                        dir / f"ground-truth-cdfs_mala_{i + 1}_{j + 1}.csv", index=False
                    )
                    pd.DataFrame(
                        {
                            "x": extend_np_array(gibbs_cat),
                            "y": extend_np_array(emp_cdf_cat),
                        }
                    ).to_csv(
                        dir / f"ground-truth-cdfs_gibbs_{i + 1}_{j + 1}.csv",
                        index=False,
                    )
                    plt.xlim([-xhr, xhr])
                else:
                    plt.plot(x, cdf, label="factor")
                    plt.plot(marginals_mala, emp_cdf, "--", label="MALA")
                    plt.plot(marginals_gibbs, emp_cdf, "--", label="Gibbs")
                    plt.xlim([xlb, xub])

                # Dump the data
                torch.save(torch.from_numpy(x), experiment_dir / "x_ground_truth.pth")
                torch.save(torch.from_numpy(f), experiment_dir / "f_ground_truth.pth")
            else:
                res_tuple = marginals(factor_to_numpy(factor["factor"]), xhr=xhr, n=n)

                if len(res_tuple) == 2:
                    x, f = res_tuple[0], res_tuple[1]

                    if type == "approximation-uniform":
                        idx = f > 0.0
                    else:
                        if type == "baseline" and key == "gmm":
                            idx = np.where((x >= -0.5) & (x <= 0.5))
                        else:
                            idx = np.where((x >= -xhr) & (x <= xhr))
                    plt.figure(fig1.number)
                    val = -np.log(f[idx])
                    plt.plot(x[idx], val)

                    pd.DataFrame(
                        {"x": extend_np_array(x[idx]), "y": extend_np_array(val)}
                    ).to_csv(dir / f"ground-truth_{i + 1}_{j + 1}.csv", index=False)

                    vlb, vub = val.min().item(), val.max().item()
                    plt.ylim([vlb - 0.025 * (vub - vlb), vub])

                    plt.figure(fig2.number)
                    x, f = res_tuple[0], res_tuple[1]
                    cdf = np.cumsum(f)
                    cdf /= cdf[-1]
                    lb_idx, ub_idx = np.argmax(cdf >= ptile), np.argmin(
                        cdf <= 1 - ptile
                    )
                    xlb, xub = x[lb_idx], x[ub_idx]

                    if type == "approximation-uniform":
                        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                        zero = torch.tensor(
                            [0.0], dtype=emp_cdf.dtype, device=emp_cdf.device
                        )
                        one = torch.tensor(
                            [1.0], dtype=emp_cdf.dtype, device=emp_cdf.device
                        )
                        emp_cdf_cat = torch.cat((zero, emp_cdf, one))
                        mala_cat = torch.cat(
                            (-one * 10 * hr, marginals_mala, one * 10 * hr)
                        )
                        gibbs_cat = torch.cat(
                            (-one * 10 * hr, marginals_gibbs, one * 10 * hr)
                        )
                        plt.plot(mala_cat, emp_cdf_cat, color=cycle[0])
                        plt.plot(gibbs_cat, emp_cdf_cat, "--", color=cycle[0])
                        pd.DataFrame(
                            {
                                "x": extend_np_array(mala_cat),
                                "y": extend_np_array(emp_cdf_cat),
                            }
                        ).to_csv(
                            dir / f"ground-truth-cdfs_mala_{i + 1}_{j + 1}.csv",
                            index=False,
                        )
                        pd.DataFrame(
                            {
                                "x": extend_np_array(gibbs_cat),
                                "y": extend_np_array(emp_cdf_cat),
                            }
                        ).to_csv(
                            dir / f"ground-truth-cdfs_gibbs_{i + 1}_{j + 1}.csv",
                            index=False,
                        )
                        plt.xlim([-xhr, xhr])
                    else:
                        plt.plot(x, cdf, label="factor")
                        plt.plot(marginals_mala, emp_cdf, "--", label="MALA")
                        plt.plot(marginals_gibbs, emp_cdf, "--", label="Gibbs")
                        plt.xlim([xlb, xub])

                    # Dump the data
                    torch.save(
                        torch.from_numpy(res_tuple[0]),
                        experiment_dir / "x_ground_truth.pth",
                    )
                    torch.save(
                        torch.from_numpy(res_tuple[1]),
                        experiment_dir / "f_ground_truth.pth",
                    )
                else:
                    plt.figure(fig1.number)
                    x1, f1 = res_tuple[0], res_tuple[1]
                    if type == "approximation-uniform":
                        idx1 = f1 > 0.0
                    else:
                        if type == "baseline" and key == "gmm":
                            idx1 = np.where((x1 >= -0.5) & (x1 <= 0.5))
                        else:
                            idx1 = np.where((x1 >= -xhr) & (x1 <= xhr))

                    val1 = -np.log(f1[idx1])
                    if j == 0:
                        plt.plot(x1[idx1], val1, label="(Inner) marginal")
                    else:
                        plt.plot(x1[idx1], val1)

                    pd.DataFrame(
                        {"x": extend_np_array(x1[idx1]), "y": extend_np_array(val1)}
                    ).to_csv(dir / f"ground-truth_{i + 1}_{j + 1}.csv", index=False)

                    x2, f2 = res_tuple[2], res_tuple[3]
                    if type == "approximation-uniform":
                        idx2 = f2 > 0.0
                    else:
                        if type == "baseline" and key == "gmm":
                            idx2 = np.where((x2 >= -0.5) & (x2 <= 0.5))
                        else:
                            idx2 = np.where((x2 >= -xhr) & (x2 <= xhr))

                    val2 = -np.log(f2[idx2])
                    if j == 0:
                        plt.plot(x2[idx2], val2, label="Outer marginal")
                    else:
                        plt.plot(x2[idx2], val2)

                    pd.DataFrame(
                        {"x": extend_np_array(x2[idx2]), "y": extend_np_array(val2)}
                    ).to_csv(
                        dir / f"ground-truth_{i + 1}_{j + 1}_orange.csv", index=False
                    )

                    vlb1, vub1 = val1.min().item(), val1.max().item()
                    vlb2, vub2 = val2.min().item(), val2.max().item()
                    vlb = min(vlb1, vlb2)
                    vub = max(vub1, vub2)
                    plt.ylim([vlb - 0.025 * (vub - vlb), vub])

                    plt.figure(fig2.number)
                    x1, f1 = res_tuple[0], res_tuple[1]
                    x2, f2 = res_tuple[2], res_tuple[3]
                    cdf1 = np.cumsum(f1)
                    cdf1 /= cdf1[-1]
                    cdf2 = np.cumsum(f2)
                    cdf2 /= cdf2[-1]
                    lb1_idx, ub1_idx = np.argmax(cdf1 >= ptile), np.argmin(
                        cdf1 <= 1 - ptile
                    )
                    x1lb, x1ub = x1[lb1_idx], x1[ub1_idx]
                    lb2_idx, ub2_idx = np.argmax(cdf2 >= ptile), np.argmin(
                        cdf2 <= 1 - ptile
                    )
                    x2lb, x2ub = x2[lb2_idx], x2[ub2_idx]
                    xlb = min(x1lb, x2lb)
                    xub = max(x1ub, x2ub)

                    if type == "approximation-uniform":
                        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                        zero = torch.tensor(
                            [0.0], dtype=emp_cdf.dtype, device=emp_cdf.device
                        )
                        one = torch.tensor(
                            [1.0], dtype=emp_cdf.dtype, device=emp_cdf.device
                        )
                        emp_cdf_inner_cat = torch.cat((zero, emp_cdf_inner, one))
                        inner_mala_cat = torch.cat(
                            (-one * 10 * hr, inner_marginals_mala, one * 10 * hr)
                        )
                        inner_gibbs_cat = torch.cat(
                            (-one * 10 * hr, inner_marginals_gibbs, one * 10 * hr)
                        )
                        emp_cdf_outer_cat = torch.cat((zero, emp_cdf_outer, one))
                        outer_mala_cat = torch.cat(
                            (-one * 10 * hr, outer_marginals_mala, one * 10 * hr)
                        )
                        outer_gibbs_cat = torch.cat(
                            (-one * 10 * hr, outer_marginals_gibbs, one * 10 * hr)
                        )
                        plt.plot(
                            inner_mala_cat,
                            emp_cdf_inner_cat,
                            label="MALA",
                            color=cycle[0],
                        )
                        plt.plot(
                            inner_gibbs_cat,
                            emp_cdf_inner_cat,
                            "--",
                            label="Gibbs",
                            color=cycle[0],
                        )
                        pd.DataFrame(
                            {
                                "x": extend_np_array(inner_mala_cat),
                                "y": extend_np_array(emp_cdf_inner_cat),
                            }
                        ).to_csv(
                            dir / f"ground-truth-cdfs_mala_inner_{i + 1}_{j + 1}.csv",
                            index=False,
                        )
                        pd.DataFrame(
                            {
                                "x": extend_np_array(inner_gibbs_cat),
                                "y": extend_np_array(emp_cdf_inner_cat),
                            }
                        ).to_csv(
                            dir / f"ground-truth-cdfs_gibbs_inner_{i + 1}_{j + 1}.csv",
                            index=False,
                        )
                        plt.plot(
                            outer_mala_cat,
                            emp_cdf_outer_cat,
                            label="MALA",
                            color=cycle[1],
                        )
                        plt.plot(
                            outer_gibbs_cat,
                            emp_cdf_outer_cat,
                            "--",
                            label="Gibbs",
                            color=cycle[1],
                        )
                        pd.DataFrame(
                            {
                                "x": extend_np_array(outer_mala_cat),
                                "y": extend_np_array(emp_cdf_outer_cat),
                            }
                        ).to_csv(
                            dir / f"ground-truth-cdfs_mala_outer_{i + 1}_{j + 1}.csv",
                            index=False,
                        )
                        pd.DataFrame(
                            {
                                "x": extend_np_array(outer_gibbs_cat),
                                "y": extend_np_array(emp_cdf_outer_cat),
                            }
                        ).to_csv(
                            dir / f"ground-truth-cdfs_gibbs_outer_{i + 1}_{j + 1}.csv",
                            index=False,
                        )
                        plt.xlim([-xhr, xhr])
                    else:
                        plt.plot(x1, cdf1, label="inner marginal")
                        plt.plot(x2, cdf2, label="outer marginal")
                        plt.plot(
                            inner_marginals_mala, emp_cdf_inner, "--", label="MALA"
                        )
                        plt.plot(
                            inner_marginals_gibbs, emp_cdf_inner, "--", label="Gibbs"
                        )
                        plt.plot(
                            outer_marginals_mala, emp_cdf_outer, "--", label="MALA"
                        )
                        plt.plot(
                            outer_marginals_gibbs, emp_cdf_outer, "--", label="Gibbs"
                        )
                        plt.xlim([xlb, xub])

                    torch.save(
                        torch.from_numpy(res_tuple[0]),
                        experiment_dir / "x_inner_ground_truth.pth",
                    )
                    torch.save(
                        torch.from_numpy(res_tuple[1]),
                        experiment_dir / "f_inner_ground_truth.pth",
                    )

                    torch.save(
                        torch.from_numpy(res_tuple[2]),
                        experiment_dir / "x_outer_ground_truth.pth",
                    )
                    torch.save(
                        torch.from_numpy(res_tuple[3]),
                        experiment_dir / "f_outer_ground_truth.pth",
                    )

            plt.figure(fig1.number)
            # if j == (cols - 1):
            #     plt.legend()
            if type == "baseline" and key == "gmm":
                plt.xlim([-0.5, 0.5])
            else:
                plt.xlim([-xhr, xhr])
            plt.tight_layout()

            plt.figure(fig2.number)
            # if j == (cols - 1):
            #     plt.legend()
            plt.tight_layout()

            print("Done!")
    plt.figure(fig1.number)
    plt.tight_layout()
    fig1.subplots_adjust(top=0.91)
    fig1.legend(
        loc="outside upper center",
        ncol=2,
    )
    plt.savefig(dir / "ground-truth.pdf")
    plt.close(fig1.number)

    plt.figure(fig2.number)
    plt.tight_layout()
    plt.savefig(dir / "ground-truth-cdfs.pdf")
    plt.close(fig2.number)

    plt.close()
    print(80 * "*")


# Computes and plots the Wasserstein distance to the ground-truth distribution over the iterations
def plot_wasserstein(type):
    # Basic sanity check
    assert type in [
        "baseline",
        "tails",
        "initialization",
        "approximation-uniform",
        "approximation-gsm",
    ]

    # Define the number of samples to use for visualizations
    if type == "tails":
        factor_viz_dict = {3: 25, 4: 25, 5: 25, 6: 25}
        product_viz_dict = {3: 25, 4: 25, 5: 25, 6: 25}
        loop_viz_dict = {3: 25, 4: 25, 5: 25, 6: 25}
        grid_viz_dict = {3: 25, 4: 25, 5: 25, 6: 25}
    elif type == "baseline":
        factor_viz_dict = {
            "normal": 25,
            "laplace": 25,
            "student-t": 25,
            "gmm": 50,
        }  # 1500
        product_viz_dict = {
            "normal": 25,
            "laplace": 25,
            "student-t": 25,
            "gmm": 25,
        }  # 50
        loop_viz_dict = {
            "normal": 25,
            "laplace": 100,
            "student-t": 25,
            "gmm": 50,
        }  # 4000
        grid_viz_dict = {
            "normal": 25,
            "laplace": 100,
            "student-t": 25,
            "gmm": 50,
        }  # 6000
    elif type == "initialization":
        factor_viz_dict = {1: 80, 5: 80, 10: 80, 15: 80}
        product_viz_dict = {1: 80, 5: 80, 10: 80, 15: 80}
        loop_viz_dict = {1: 80, 5: 80, 10: 80, 15: 80}
        grid_viz_dict = {1: 80, 5: 80, 10: 80, 15: 80}
    elif type == "approximation-uniform":
        factor_viz_dict = {512: 100, 1024: 100, 1536: 100, 2048: 100}
        product_viz_dict = {512: 100, 1024: 100, 1536: 100, 2048: 100}
        loop_viz_dict = {512: 100, 1024: 100, 1536: 100, 2048: 100}
        grid_viz_dict = {512: 100, 1024: 100, 1536: 100, 2048: 100}
    else:
        factor_viz_dict = {256: 100, 512: 100, 1024: 100, 2048: 100}
        product_viz_dict = {256: 100, 512: 100, 1024: 100, 2048: 100}
        loop_viz_dict = {256: 100, 512: 100, 1024: 100, 2048: 100}
        grid_viz_dict = {256: 100, 512: 100, 1024: 100, 2048: 100}
    viz_dict = {
        "factor": factor_viz_dict,
        "product": product_viz_dict,
        "loop": loop_viz_dict,
        "grid": grid_viz_dict,
    }

    # Read the model configurations
    if type == "tails":
        df_list = dict(
            OmegaConf.load("sampling/experiments/configs/sampling/baseline/tails.yaml")
        )["df_list"]
        factor_configs = {df: df for df in df_list}
    elif type == "baseline":
        factor_configs = dict(
            OmegaConf.load(
                "sampling/experiments/configs/sampling/baseline/baseline.yaml"
            )
        )
    elif type == "initialization":
        norm_list = dict(
            OmegaConf.load(
                "sampling/experiments/configs/sampling/baseline/initialization.yaml"
            )
        )["norm_list"]
        factor_configs = {norm: norm for norm in norm_list}
    elif type == "approximation-uniform":
        num_components_list = dict(
            OmegaConf.load(
                "sampling/experiments/configs/sampling/baseline/approximation_uniform.yaml"
            )
        )["num_components_list"]
        factor_configs = {
            num_components: num_components for num_components in num_components_list
        }
    else:
        num_components_list = dict(
            OmegaConf.load(
                "sampling/experiments/configs/sampling/baseline/approximation_gsm.yaml"
            )
        )["num_components_list"]
        factor_configs = {
            num_components: num_components for num_components in num_components_list
        }

    # Define the topologies
    topology_list = ["factor", "product", "loop", "grid"]

    # Get the data directory
    root_dir = get_data_dir()

    # Get the output directory
    if type == "tails":
        dir = root_dir / "experiments" / "sampling" / "baseline" / "tails"
    elif type == "baseline":
        dir = root_dir / "experiments" / "sampling" / "baseline" / "baseline"
    elif type == "initialization":
        dir = root_dir / "experiments" / "sampling" / "baseline" / "initialization"
    elif type == "approximation-uniform":
        dir = (
            root_dir
            / "experiments"
            / "sampling"
            / "baseline"
            / "approximation"
            / "uniform"
        )
    else:
        dir = (
            root_dir / "experiments" / "sampling" / "baseline" / "approximation" / "gsm"
        )

    # Compute the Wasserstein distances
    print()
    print(80 * "*")
    if type == "tails":
        print(
            "Running Wasserstein computations for marginals in the baseline tail experiments"
        )
    elif type == "baseline":
        print(
            "Running Wasserstein computations for marginals in the baseline experiments"
        )
    elif type == "initialization":
        print(
            "Running Wasserstein computations for marginals in the baseline initialization experiments"
        )
    elif type == "approximation-uniform":
        print(
            "Running Wasserstein computations for marginals in the baseline uniform approximation experiments"
        )
    else:
        print(
            "Running Wasserstein computations for marginals in the baseline gsm approximation experiments"
        )

    # Run the experiments
    rows, cols = len(topology_list), len(factor_configs)
    plt.figure(figsize=(4.5 * cols, 4.5 * rows))
    for i, topology in enumerate(topology_list):
        for j, key in enumerate(factor_configs.keys()):
            print()
            if type == "tails":
                print(
                    f"Computing for {topology} graph and student-t factors with df = {key}..."
                )
            elif type == "baseline":
                print(f"Computing for {topology} graph and {key} factors...")
            elif type == "initialization":
                print(
                    f"Computing for {topology} graph and student-t factors with init norm = {key}..."
                )
            else:
                print(
                    f"Computing for {topology} graph and GMM factors with {key} components..."
                )

            idx = i * cols + j + 1
            plt.subplot(rows, cols, idx)

            if i == 0:
                if type == "tails":
                    plt.title(rf"$\nu = {key}$")
                elif type == "baseline":
                    plt.title(capitalize(key))
                elif type == "initialization":
                    plt.title(rf"$c = {key}$")
                else:
                    plt.title(f"${key}$")

            if j == 0:
                plt.ylabel(capitalize(topology))

            # Get the dir of the current experiment
            experiment_dir = dir / topology / f"{key}"

            # Get the number of samples
            num_samples = viz_dict[topology][key] + 1

            # Load the MALA and Gibbs samples
            samples_mala = torch.load(
                experiment_dir / "samples_mala.pth", weights_only=True
            )[:num_samples]
            samples_gibbs = torch.load(
                experiment_dir / "samples_gibbs.pth", weights_only=True
            )[:num_samples]

            # Basic sanity checks
            assert is_finite(samples_mala)
            assert is_finite(samples_gibbs)

            # Load the ground-truth data
            if topology != "grid":
                # Compute the marginals of the samples
                marginals_mala = compute_sample_marginals(topology, samples_mala)
                marginals_gibbs = compute_sample_marginals(topology, samples_gibbs)

                # Sort the marginals along the chain dimension
                marginals_mala = torch.sort(marginals_mala, dim=1)[0]
                marginals_gibbs = torch.sort(marginals_gibbs, dim=1)[0]

                # Construct empirical CDF tensor
                emp_cdf = torch.cumsum(
                    torch.ones_like(marginals_mala[0]).to(dtype=torch.float64)
                    / marginals_mala.shape[1],
                    dim=0,
                )

                # Load the ground-truth marginals
                device = marginals_mala.device
                x_gt = torch.load(
                    experiment_dir / "x_ground_truth.pth", weights_only=True
                ).to(device)
                f_gt = torch.load(
                    experiment_dir / "f_ground_truth.pth", weights_only=True
                ).to(device)

                # Basic sanity checks
                assert is_finite(x_gt)
                assert is_finite(f_gt)

                # Compute their CDF
                cdf = torch.cumsum(f_gt, dim=0)
                cdf /= torch.max(cdf)

                # Compute the Wasserstein-1 distances over the iterations for MALA and Gibbs
                w1_mala = torch.zeros((num_samples,), dtype=torch.float64)
                w1_gibbs = torch.zeros((num_samples,), dtype=torch.float64)

                for k in range(num_samples):
                    w1_mala[k] = wasserstein_1_distance(
                        x_gt, zoh(marginals_mala[k], emp_cdf, x_gt), cdf
                    )
                    w1_gibbs[k] = wasserstein_1_distance(
                        x_gt, zoh(marginals_gibbs[k], emp_cdf, x_gt), cdf
                    )

                # Plot the results
                cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                plt.plot(torch.log10(w1_mala).cpu().numpy(), color=cycle[0])
                plt.plot(torch.log10(w1_gibbs).cpu().numpy(), "--", color=cycle[0])
                plt.xlim([0, num_samples - 1])
            else:
                # Compute the marginals of the samples
                inner_marginals_mala, outer_marginals_mala = compute_sample_marginals(
                    topology, samples_mala
                )
                inner_marginals_gibbs, outer_marginals_gibbs = compute_sample_marginals(
                    topology, samples_gibbs
                )

                # Sort the marginals along the chain dimension
                inner_marginals_mala = torch.sort(inner_marginals_mala, dim=1)[0]
                outer_marginals_mala = torch.sort(outer_marginals_mala, dim=1)[0]
                inner_marginals_gibbs = torch.sort(inner_marginals_gibbs, dim=1)[0]
                outer_marginals_gibbs = torch.sort(outer_marginals_gibbs, dim=1)[0]

                # Construct empirical CDF tensors
                inner_emp_cdf = torch.cumsum(
                    torch.ones_like(inner_marginals_mala[0]).to(dtype=torch.float64)
                    / inner_marginals_mala.shape[1],
                    dim=0,
                )
                outer_emp_cdf = torch.cumsum(
                    torch.ones_like(outer_marginals_mala[0]).to(dtype=torch.float64)
                    / outer_marginals_mala.shape[1],
                    dim=0,
                )

                # Load the ground-truth marginals
                device = inner_marginals_mala.device
                x_inner_gt = torch.load(
                    experiment_dir / "x_inner_ground_truth.pth", weights_only=True
                ).to(device)
                f_inner_gt = torch.load(
                    experiment_dir / "f_inner_ground_truth.pth", weights_only=True
                ).to(device)

                x_outer_gt = torch.load(
                    experiment_dir / "x_outer_ground_truth.pth", weights_only=True
                ).to(device)
                f_outer_gt = torch.load(
                    experiment_dir / "f_outer_ground_truth.pth", weights_only=True
                ).to(device)

                # Basic sanity checks
                assert is_finite(x_inner_gt)
                assert is_finite(f_inner_gt)

                assert is_finite(x_outer_gt)
                assert is_finite(f_outer_gt)

                # Compute their CDF
                inner_cdf = torch.cumsum(f_inner_gt, dim=0)
                inner_cdf /= torch.max(inner_cdf)

                outer_cdf = torch.cumsum(f_outer_gt, dim=0)
                outer_cdf /= torch.max(outer_cdf)

                # Compute the Wasserstein-1 distances over the iterations for MALA and Gibbs
                w1_inner_mala = torch.zeros((num_samples,), dtype=torch.float64)
                w1_outer_mala = torch.zeros((num_samples,), dtype=torch.float64)
                w1_inner_gibbs = torch.zeros((num_samples,), dtype=torch.float64)
                w1_outer_gibbs = torch.zeros((num_samples,), dtype=torch.float64)

                for k in range(num_samples):
                    w1_inner_mala[k] = wasserstein_1_distance(
                        x_inner_gt,
                        zoh(inner_marginals_mala[k], inner_emp_cdf, x_inner_gt),
                        inner_cdf,
                    )
                    w1_outer_mala[k] = wasserstein_1_distance(
                        x_outer_gt,
                        zoh(
                            outer_marginals_mala[k],
                            outer_emp_cdf,
                            x_outer_gt,
                            num_chunks=10,
                        ),
                        outer_cdf,
                    )

                    w1_inner_gibbs[k] = wasserstein_1_distance(
                        x_inner_gt,
                        zoh(inner_marginals_gibbs[k], inner_emp_cdf, x_inner_gt),
                        inner_cdf,
                    )
                    w1_outer_gibbs[k] = wasserstein_1_distance(
                        x_outer_gt,
                        zoh(
                            outer_marginals_gibbs[k],
                            outer_emp_cdf,
                            x_outer_gt,
                            num_chunks=10,
                        ),
                        outer_cdf,
                    )

                # Plot the results
                cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                plt.plot(torch.log10(w1_inner_mala), color=cycle[0])
                plt.plot(torch.log10(w1_inner_gibbs), "--", color=cycle[0])
                plt.plot(torch.log10(w1_outer_mala), color=cycle[1])
                plt.plot(torch.log10(w1_outer_gibbs), "--", color=cycle[1])
                plt.xlim([0, num_samples - 1])

            plt.tight_layout()
            print("Done!")
    plt.savefig(dir / "wasserstein.pdf")
    plt.close()
    print(80 * "*")


# Computes and plots average ACF
def plot_acf(type, burn_in=7500, subsample=1, num_lags_viz=30):
    # Basic sanity checks
    assert type in [
        "baseline",
        "tails",
        "initialization",
        "approximation-uniform",
        "approximation-gsm",
    ]
    assert burn_in > 0
    assert subsample > 0
    assert num_lags_viz > 0

    # Read the model configurations
    if type == "tails":
        df_list = dict(
            OmegaConf.load("sampling/experiments/configs/sampling/baseline/tails.yaml")
        )["df_list"]
        factor_configs = {df: df for df in df_list}
    elif type == "baseline":
        factor_configs = dict(
            OmegaConf.load(
                "sampling/experiments/configs/sampling/baseline/baseline.yaml"
            )
        )
    elif type == "initialization":
        norm_list = dict(
            OmegaConf.load(
                "sampling/experiments/configs/sampling/baseline/initialization.yaml"
            )
        )["norm_list"]
        factor_configs = {norm: norm for norm in norm_list}
    elif type == "approximation-uniform":
        num_components_list = dict(
            OmegaConf.load(
                "sampling/experiments/configs/sampling/baseline/approximation_uniform.yaml"
            )
        )["num_components_list"]
        factor_configs = {
            num_components: num_components for num_components in num_components_list
        }
    else:
        num_components_list = dict(
            OmegaConf.load(
                "sampling/experiments/configs/sampling/baseline/approximation_gsm.yaml"
            )
        )["num_components_list"]
        factor_configs = {
            num_components: num_components for num_components in num_components_list
        }

    # Define the topologies
    topology_list = ["factor", "product", "loop", "grid"]

    # Get the data directory
    root_dir = get_data_dir()

    # Get the output directory
    if type == "tails":
        dir = root_dir / "experiments" / "sampling" / "baseline" / "tails"
    elif type == "baseline":
        dir = root_dir / "experiments" / "sampling" / "baseline" / "baseline"
    elif type == "initialization":
        dir = root_dir / "experiments" / "sampling" / "baseline" / "initialization"
    elif type == "approximation-uniform":
        dir = (
            root_dir
            / "experiments"
            / "sampling"
            / "baseline"
            / "approximation"
            / "uniform"
        )
    else:
        dir = (
            root_dir / "experiments" / "sampling" / "baseline" / "approximation" / "gsm"
        )

    # Compute the ACFs
    print()
    print(80 * "*")
    if type == "tails":
        print("Running ACF computations for marginals in the baseline tail experiments")
    elif type == "baseline":
        print("Running ACF computations for marginals in the baseline experiments")
    elif type == "initialization":
        print(
            "Running ACF computations for marginals in the baseline initialization experiments"
        )
    elif type == "approximation-uniform":
        print(
            "Running ACF computations for marginals in the baseline uniform approximation experiments"
        )
    else:
        print(
            "Running ACF computations for marginals in the baseline gsm approximation experiments"
        )
    print(80 * "*")

    # Run the experiments
    rows, cols = len(topology_list), len(factor_configs)
    plt.figure(figsize=(4.5 * cols, 4.5 * rows))
    for i, topology in enumerate(topology_list):
        for j, key in enumerate(factor_configs.keys()):
            print()
            if type == "tails":
                print(
                    f"Computing for {topology} graph and student-t factors with df = {key}..."
                )
            elif type == "baseline":
                print(f"Computing for {topology} graph and {key} factors...")
            elif type == "initialization":
                print(
                    f"Computing for {topology} graph and student-t factors with init norm = {key}..."
                )
            else:
                print(
                    f"Computing for {topology} graph and GMM factors with {key} components..."
                )

            idx = i * cols + j + 1
            plt.subplot(rows, cols, idx)

            if i == 0:
                if type == "tails":
                    plt.title(rf"$\nu = {key}$")
                elif type == "baseline":
                    plt.title(capitalize(key))
                elif type == "initialization":
                    plt.title(rf"$c = {key}$")
                else:
                    plt.title(f"${key}$")

            if j == 0:
                plt.ylabel(capitalize(topology))

            # Get the dir of the current experiment
            experiment_dir = dir / topology / f"{key}"

            # Load the MALA and Gibbs samples
            samples_mala = torch.load(
                experiment_dir / "samples_mala.pth",
                weights_only=True,
                map_location=torch.device("cpu"),
            )[burn_in + 1 :, ::subsample]
            samples_gibbs = torch.load(
                experiment_dir / "samples_gibbs.pth",
                weights_only=True,
                map_location=torch.device("cpu"),
            )[burn_in + 1 :, ::subsample]

            # Basic sanity checks
            assert is_finite(samples_mala)
            assert is_finite(samples_gibbs)

            if topology != "grid":
                # Compute the marginals of the samples
                marginals_mala = compute_sample_marginals(topology, samples_mala)
                marginals_gibbs = compute_sample_marginals(topology, samples_gibbs)

                # Compute the output shape for the ACF computations
                num_chains = marginals_mala.shape[1]
                # NOTE: A larger number of lags is required for GMM potentials to make the effective sample size
                # computations accurate
                num_lags_arg = None if type == "baseline" and key == "GMM" else 1250
                nlags = acf(marginals_mala[:, 0].cpu(), nlags=num_lags_arg).size

                # Construct numpy arrays to store the outputs
                acf_mala = np.zeros((num_chains, nlags))
                acf_gibbs = np.zeros((num_chains, nlags))

                # Compute the ACFs
                for k in range(num_chains):
                    acf_mala[k] = acf(marginals_mala[:, k].cpu(), nlags=num_lags_arg)
                    acf_gibbs[k] = acf(marginals_gibbs[:, k].cpu(), nlags=num_lags_arg)

                # Ignore the zero-lag entry
                acf_mala = acf_mala[:, 1:]
                acf_gibbs = acf_gibbs[:, 1:]

                # Compute the statistics
                acf_mala_mean = acf_mala.mean(axis=0)
                acf_gibbs_mean = acf_gibbs.mean(axis=0)
                acf_mala_std = acf_mala.std(axis=0)
                acf_gibbs_std = acf_gibbs.std(axis=0)

                # Plot the results
                cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                lags = np.array([i + 1 for i in range(nlags - 1)])
                plt.fill_between(
                    lags,
                    acf_mala_mean - acf_mala_std,
                    acf_mala_mean + acf_mala_std,
                    alpha=0.5,
                    color=cycle[0],
                )
                plt.fill_between(
                    lags,
                    acf_gibbs_mean - acf_gibbs_std,
                    acf_gibbs_mean + acf_gibbs_std,
                    alpha=0.5,
                    color=cycle[0],
                )
                plt.scatter(lags, acf_mala_mean, s=60, edgecolors=cycle[0])
                plt.scatter(
                    lags, acf_gibbs_mean, s=60, edgecolors=cycle[0], facecolors="white"
                )
                plt.xlim([1, num_lags_viz])

                pd.DataFrame(
                    {
                        "x": lags,
                        "y": acf_mala_mean,
                        "upper": acf_mala_mean + acf_mala_std,
                        "lower": acf_mala_mean - acf_mala_std,
                    }
                ).to_csv(dir / f"acf_mala_{i + 1}_{j + 1}.csv", index=False)

                pd.DataFrame(
                    {
                        "x": lags,
                        "y": acf_gibbs_mean,
                        "upper": acf_gibbs_mean + acf_gibbs_std,
                        "lower": acf_gibbs_mean - acf_gibbs_std,
                    }
                ).to_csv(dir / f"acf_gibbs_{i + 1}_{j + 1}.csv", index=False)

                # Compute the effective sample sizes
                # Ignore all autocorrelation values below 0.05
                acf_mala[acf_mala < 0.05] = 0.0
                acf_gibbs[acf_gibbs < 0.05] = 0.0

                # Compute the ratio per chain
                ratios_mala = 1.0 / (1.0 + 2.0 * np.sum(acf_mala, axis=1))
                ratios_gibbs = 1.0 / (1.0 + 2.0 * np.sum(acf_gibbs, axis=1))

                # Report means and stds
                print(
                    f"Neff MALA  = {ratios_mala.mean():5.3f} +- {ratios_mala.std():.3f}"
                )
                print(
                    f"Neff Gibbs = {ratios_gibbs.mean():5.3f} +- {ratios_gibbs.std():.3f}"
                )
            else:
                # Compute the marginals of the samples
                inner_marginals_mala, outer_marginals_mala = compute_sample_marginals(
                    topology, samples_mala
                )
                inner_marginals_gibbs, outer_marginals_gibbs = compute_sample_marginals(
                    topology, samples_gibbs
                )

                # Compute the output shape for the ACF computations of the inner marginals
                num_chains_inner = inner_marginals_mala.shape[1]
                # NOTE: A larger number of lags is required for GMM potentials to make the effective sample size
                # computations accurate
                num_lags_arg = None if type == "baseline" and key == "GMM" else 1250
                nlags = acf(inner_marginals_mala[:, 0].cpu(), nlags=num_lags_arg).size

                # Construct numpy arrays to store the outputs of the inner marginals
                acf_mala_inner = np.zeros((num_chains_inner, nlags))
                acf_gibbs_inner = np.zeros((num_chains_inner, nlags))

                # Compute the ACFs of the inner marginals
                for k in range(num_chains_inner):
                    acf_mala_inner[k] = acf(
                        inner_marginals_mala[:, k].cpu(), nlags=num_lags_arg
                    )
                    acf_gibbs_inner[k] = acf(
                        inner_marginals_gibbs[:, k].cpu(), nlags=num_lags_arg
                    )

                # Construct numpy arrays to store the outputs of the outer marginals
                num_chains_outer = outer_marginals_mala.shape[1]
                acf_mala_outer = np.zeros((num_chains_outer, nlags))
                acf_gibbs_outer = np.zeros((num_chains_outer, nlags))

                # Compute the ACFs of the outer marginals
                for k in range(num_chains_outer):
                    acf_mala_outer[k] = acf(
                        outer_marginals_mala[:, k].cpu(), nlags=num_lags_arg
                    )
                    acf_gibbs_outer[k] = acf(
                        outer_marginals_gibbs[:, k].cpu(), nlags=num_lags_arg
                    )

                # Ignore the zero-lag entry
                acf_mala_inner = acf_mala_inner[:, 1:]
                acf_gibbs_inner = acf_gibbs_inner[:, 1:]
                acf_mala_outer = acf_mala_outer[:, 1:]
                acf_gibbs_outer = acf_gibbs_outer[:, 1:]

                # Compute the statistics
                acf_mala_inner_mean = acf_mala_inner.mean(axis=0)
                acf_gibbs_inner_mean = acf_gibbs_inner.mean(axis=0)
                acf_mala_inner_std = acf_mala_inner.std(axis=0)
                acf_gibbs_inner_std = acf_gibbs_inner.std(axis=0)

                acf_mala_outer_mean = acf_mala_outer.mean(axis=0)
                acf_gibbs_outer_mean = acf_gibbs_outer.mean(axis=0)
                acf_mala_outer_std = acf_mala_outer.std(axis=0)
                acf_gibbs_outer_std = acf_gibbs_outer.std(axis=0)

                # Plot the results
                cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                lags = np.array([i + 1 for i in range(nlags - 1)])
                plt.fill_between(
                    lags,
                    acf_mala_inner_mean - acf_mala_inner_std,
                    acf_mala_inner_mean + acf_mala_inner_std,
                    alpha=0.5,
                    color=cycle[0],
                )
                plt.fill_between(
                    lags,
                    acf_gibbs_inner_mean - acf_gibbs_inner_std,
                    acf_gibbs_inner_mean + acf_gibbs_inner_std,
                    alpha=0.5,
                    color=cycle[0],
                )
                plt.scatter(lags, acf_mala_inner_mean, s=60, edgecolors=cycle[0])
                plt.scatter(
                    lags,
                    acf_gibbs_inner_mean,
                    s=60,
                    edgecolors=cycle[0],
                    facecolors="white",
                )
                plt.xlim([1, num_lags_viz])

                plt.fill_between(
                    lags,
                    acf_mala_outer_mean - acf_mala_outer_std,
                    acf_mala_outer_mean + acf_mala_outer_std,
                    alpha=0.5,
                    color=cycle[1],
                )
                plt.fill_between(
                    lags,
                    acf_gibbs_outer_mean - acf_gibbs_outer_std,
                    acf_gibbs_outer_mean + acf_gibbs_outer_std,
                    alpha=0.5,
                    color=cycle[1],
                )
                plt.scatter(lags, acf_mala_outer_mean, s=60, edgecolors=cycle[1])
                plt.scatter(
                    lags,
                    acf_gibbs_outer_mean,
                    s=60,
                    edgecolors=cycle[1],
                    facecolors="white",
                )
                plt.xlim([1, num_lags_viz])

                pd.DataFrame(
                    {
                        "x": lags,
                        "y": acf_mala_inner_mean,
                        "upper": acf_mala_inner_mean + acf_mala_inner_std,
                        "lower": acf_mala_inner_mean - acf_mala_inner_std,
                    }
                ).to_csv(dir / f"acf_mala_inner_{i + 1}_{j + 1}.csv", index=False)

                pd.DataFrame(
                    {
                        "x": lags,
                        "y": acf_gibbs_inner_mean,
                        "upper": acf_gibbs_inner_mean + acf_gibbs_inner_std,
                        "lower": acf_gibbs_inner_mean - acf_gibbs_inner_std,
                    }
                ).to_csv(dir / f"acf_gibbs_inner_{i + 1}_{j + 1}.csv", index=False)

                pd.DataFrame(
                    {
                        "x": lags,
                        "y": acf_mala_outer_mean,
                        "upper": acf_mala_outer_mean + acf_mala_outer_std,
                        "lower": acf_mala_outer_mean - acf_mala_outer_std,
                    }
                ).to_csv(dir / f"acf_mala_outer_{i + 1}_{j + 1}.csv", index=False)

                pd.DataFrame(
                    {
                        "x": lags,
                        "y": acf_gibbs_outer_mean,
                        "upper": acf_gibbs_outer_mean + acf_gibbs_outer_std,
                        "lower": acf_gibbs_outer_mean - acf_gibbs_outer_std,
                    }
                ).to_csv(dir / f"acf_gibbs_outer_{i + 1}_{j + 1}.csv", index=False)

                # Compute the effective sample sizes
                # Ignore all autocorrelation values below 0.05
                acf_mala_inner[acf_mala_inner < 0.05] = 0.0
                acf_gibbs_inner[acf_gibbs_inner < 0.05] = 0.0

                acf_mala_outer[acf_mala_outer < 0.05] = 0.0
                acf_gibbs_outer[acf_gibbs_outer < 0.05] = 0.0

                # Compute the ratio per chain
                ratios_mala_inner = 1.0 / (1.0 + 2.0 * np.sum(acf_mala_inner, axis=1))
                ratios_gibbs_inner = 1.0 / (1.0 + 2.0 * np.sum(acf_gibbs_inner, axis=1))

                ratios_mala_outer = 1.0 / (1.0 + 2.0 * np.sum(acf_mala_outer, axis=1))
                ratios_gibbs_outer = 1.0 / (1.0 + 2.0 * np.sum(acf_gibbs_outer, axis=1))

                # Report means and stds
                print(
                    f"Neff MALA  inner = {ratios_mala_inner.mean():5.3f} +- {ratios_mala_inner.std():.3f}"
                )
                print(
                    f"Neff Gibbs inner = {ratios_gibbs_inner.mean():5.3f} +- {ratios_gibbs_inner.std():.3f}"
                )
                print(
                    f"Neff MALA  outer = {ratios_mala_outer.mean():5.3f} +- {ratios_mala_outer.std():.3f}"
                )
                print(
                    f"Neff Gibbs outer = {ratios_gibbs_outer.mean():5.3f} +- {ratios_gibbs_outer.std():.3f}"
                )

            plt.tight_layout()
            print("Done!")
    plt.savefig(dir / "acf.pdf")
    plt.close()
    print(80 * "*")


if __name__ == "__main__":
    # Load global plotting parameters
    init_matplotlib()

    # Run the ground-truth computations
    plot_ground_truth_marginals("baseline")
    plot_ground_truth_marginals("initialization")
    plot_ground_truth_marginals("approximation-uniform")
    plot_ground_truth_marginals("approximation-gsm")

    # Create the Wasserstein plots
    plot_wasserstein("baseline")
    plot_wasserstein("initialization")
    plot_wasserstein("approximation-gsm")

    # Create the ACF plots
    plot_acf("baseline")
    plot_acf("approximation-gsm")
