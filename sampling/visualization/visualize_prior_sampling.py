import gc
import os
from functools import partial

import imageio
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf
from tqdm import tqdm

from sampling.experiments.priors import pot_gmm_gen
from sampling.samplers import potential
from sampling.util import (
    init_matplotlib,
    get_data_dir,
    wasserstein_1_samples,
    is_finite, extend_np_array,
)


# Utility function that computes the runtimes of the prior sampling experiments
def compute_runtimes():
    # Get the factors and patch sizes
    factor_list, patch_size_list, _ = get_configurations()

    # Get the output directory
    dir = get_output_dir()

    def convert_nanoseconds_to_hms(ns: int) -> str:
        # Convert nanoseconds to integer total seconds
        total_seconds = ns // 1_000_000_000

        # Compute hours, minutes, and seconds
        hours = total_seconds // 3600
        remaining = total_seconds % 3600
        minutes = remaining // 60
        seconds = remaining % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    print()
    print(80 * "*")
    print("Running runtime computations")
    print(80 * "*")
    for i, patch_size in enumerate(patch_size_list):
        for j, factor_name in enumerate(factor_list):
            print()
            print(
                f"Computing for {factor_name} factors and patch size {patch_size} x {patch_size}..."
            )

            # Get the dir of the current experiment
            experiment_dir = dir / factor_name / f"{patch_size}"

            if patch_size != 96 or factor_name != "gmm":
                times_mala = torch.load(
                    experiment_dir / "times_mala.pth", weights_only=True
                )[:-1]
                times_mala[1:] = times_mala[1:] - times_mala[1]
                print("MALA:", convert_nanoseconds_to_hms(times_mala[-1].item()))

            times_gibbs = torch.load(
                experiment_dir / "times_gibbs.pth", weights_only=True
            )[:-1]
            times_gibbs[1:] = times_gibbs[1:] - times_gibbs[1]

            print("Gibbs:", convert_nanoseconds_to_hms(times_gibbs[-1].item()))
    print("\nDone!")
    print(80 * "*")


########################################################################################################################
# Plotting of sampling results
########################################################################################################################
# Utility function that defines the factors, patch sizes and considered models
def get_configurations(dtype=torch.float64, device="cuda"):
    # Define the factors
    factor_list = ["normal", "laplace", "student-t", "gmm"]

    # Define the patch sizes
    patch_size_list = [12, 24, 48, 96]

    # Define the considered models
    # Normal configuration
    mean = torch.tensor(0.0, dtype=dtype, device=device)
    std = torch.tensor(0.4, dtype=dtype, device=device)
    normal = torch.distributions.Normal(mean, std)
    pot_normal = partial(potential, normal)

    # Laplace configuration
    loc = torch.tensor(0.0, dtype=dtype, device=device)
    b = torch.tensor(0.3, dtype=dtype, device=device)
    laplace = torch.distributions.Laplace(loc, b)
    pot_laplace = partial(potential, laplace)

    # Student-t configuration
    df = torch.tensor(6.0, dtype=dtype, device=device)
    student_t = torch.distributions.StudentT(df)
    pot_student_t = partial(potential, student_t)

    # GMM
    weights = torch.tensor(
        [
            [
                0.1058,
                0.0054,
                0.0886,
                0.0733,
                0.0667,
                0.0077,
                0.0591,
                0.3213,
                0.2721,
            ],
            [
                0.1058,
                0.0054,
                0.0886,
                0.0733,
                0.0667,
                0.0077,
                0.0591,
                0.3213,
                0.2721,
            ],
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

    # Construct the model configuration dictionaries
    potentials_dict = {
        "normal": pot_normal,
        "laplace": pot_laplace,
        "student-t": pot_student_t,
        "gmm": pot_gmm,
    }

    return factor_list, patch_size_list, potentials_dict


# Utility function that defines the output directory
def get_output_dir():
    # Get the data directory
    root_dir = get_data_dir()

    return root_dir / "experiments" / "sampling" / "prior"


# Utility function that makes a collage of the samples for a given size
def plot_collage(patch_size, num_samples):
    # Get the factors and patch sizes
    factor_list, patch_size_list, _ = get_configurations()

    # Basic sanity checks
    assert patch_size in patch_size_list
    assert num_samples > 0

    # Get the output directory
    dir = get_output_dir()

    # Create the collage
    print()
    print(80 * "*")
    print(f"Creating collage for patch_size = {patch_size}")
    print(80 * "*")

    # Computing the visualization ranges
    vmin, vmax = 10.0, -10.0
    for factor_name in factor_list:
        # Get the dir of the current factor
        experiment_dir = dir / factor_name / f"{patch_size}"

        # Load the Gibbs samples
        samples_gibbs = torch.load(
            experiment_dir / "samples_gibbs.pth", weights_only=True
        )

        # Update the ranges
        vmin = min(vmin, samples_gibbs.min().item())
        vmax = max(vmax, samples_gibbs.max().item())

    # Visualize the samples
    rows, cols = len(factor_list), num_samples
    plt.figure(figsize=(4.5 * cols, 4.5 * rows))
    for i, factor_name in enumerate(factor_list):
        # Get the dir of the current factor
        experiment_dir = dir / factor_name / f"{patch_size}"

        # Load the Gibbs samples
        samples_gibbs = torch.load(
            experiment_dir / "samples_gibbs.pth", weights_only=True
        )
        dtype = samples_gibbs.dtype
        device = samples_gibbs.device

        # Visualize the samples
        idx = 1 + i * num_samples

        # Extract samples for paper
        samples_gibbs_chunk = samples_gibbs[:num_samples, 0, :, :].cpu()
        vmin, vmax = samples_gibbs_chunk.min().item(), samples_gibbs_chunk.max().item()
        print("vmin =", vmin, "vmax =", vmax)
        samples_gibbs_chunk = torch.round(
            (samples_gibbs_chunk - vmin) / (vmax - vmin) * 255
        ).byte()
        for j in range(num_samples):
            imageio.imwrite(
                dir / f"collage_{i + 1}_{j + 1}.png", samples_gibbs_chunk[j].numpy()
            )

            plt.subplot(rows, cols, idx + j)
            plt.imshow(
                # samples_gibbs[j, 0, :, :].cpu(), vmin=vmin, vmax=vmax, cmap="gray"
                samples_gibbs[j, 0, :, :].cpu(),
                cmap="gray",
            )
            plt.xticks([])
            plt.yticks([])
            if j == 0:
                plt.ylabel(factor_name)

    plt.tight_layout()
    plt.savefig(dir / f"collage-{patch_size}.pdf")
    plt.close()

    # Set up convolutional layer that contains all MRF filters
    conv = torch.nn.Conv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=(3, 3),
        bias=False,
        dtype=dtype,
        device=device,
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

    plt.figure(figsize=(8, 8))

    # Load the dataset samples
    experiment_dir = dir
    samples_dataset = torch.load(
        experiment_dir / "samples_dataset.pth", weights_only=True
    )

    # Compute the marginals
    D_samples_data = conv(samples_dataset).reshape(-1)
    D_samples_data = torch.sort(D_samples_data, dim=0)[0]

    print(D_samples_data.min(), D_samples_data.max())

    # Plot the histograms
    D_counts, D_bins = torch.histogram(D_samples_data.cpu(), bins=1000, density=True)
    log_vals = -torch.log(D_counts)
    plt.plot(D_bins[:-1], log_vals - log_vals.min(), label="dataset")
    pd.DataFrame({"x": D_bins[:-1], "y": log_vals - log_vals.min()}).to_csv(
        dir / f"dx_marginals_dataset.csv", index=False
    )

    scalings = {
        "normal": 3.6545774260173203,
        "laplace": 3.7761423265910774,
        "student-t": 13.117132132132133,
        "gmm": 0.5803,
    }
    for factor_name in factor_list:
        if factor_name not in ["normal", "laplace", "student-t", "gmm"]:
            continue

        # Get the dir of the current factor
        experiment_dir = dir / factor_name / f"{patch_size}"

        # Load the Gibbs samples
        samples_gibbs = torch.load(
            experiment_dir / "samples_gibbs.pth", weights_only=True
        )

        # Compute the marginals
        D_samples = conv(samples_gibbs / scalings[factor_name]).reshape(-1)

        print(D_samples.min().item(), D_samples.max().item())

        # Plot the histograms
        D_counts, D_bins = torch.histogram(D_samples.cpu(), bins=1000, density=True)
        log_vals = -torch.log(D_counts)
        plt.plot(D_bins[:-1], -torch.log(D_counts) - log_vals.min(), label=factor_name)
        pd.DataFrame(
            {"x": D_bins[:-1], "y": -torch.log(D_counts) - log_vals.min()}
        ).to_csv(dir / f"dx_marginals_{factor_name}.csv", index=False)

    plt.xlim([-1.0, 1.0])
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(dir / f"D-{patch_size}-marginals.pdf")
    plt.close()

    print("Done!")
    print(80 * "*")


# Plots the generated MALA and Gibbs samples
def plot_samples(num_samples):
    # Basic sanity check
    assert num_samples > 0

    # Get the factors and patch sizes
    factor_list, patch_size_list, _ = get_configurations()

    # Get the output directory
    dir = get_output_dir()

    # Compute the ground-truth
    print()
    print(80 * "*")
    print("Creating sample plots")
    print(80 * "*")

    # Run the experiments
    rows, cols = num_samples, num_samples
    fig_mala = plt.figure(figsize=(1 * cols, 1 * rows))
    fig_gibbs = plt.figure(figsize=(1 * cols, 1 * rows))
    for patch_size in patch_size_list:
        for factor_name in factor_list:
            print()
            print(
                f"Creating sample plots for {factor_name} factors and patch size {patch_size} x {patch_size}..."
            )

            # Get the dir of the current experiment
            experiment_dir = dir / factor_name / f"{patch_size}"

            # Load the MALA and Gibbs samples
            condition = patch_size != 96 or factor_name != "gmm"
            if condition:
                samples_mala = torch.load(
                    experiment_dir / "samples_mala.pth", weights_only=True
                )
            samples_gibbs = torch.load(
                experiment_dir / "samples_gibbs.pth", weights_only=True
            )

            # Load the full tensors
            if condition:
                f_mala_full = torch.load(
                    experiment_dir / "f_mala.pth",
                    weights_only=True,
                    map_location="cpu",
                )
            f_gibbs_full = torch.load(experiment_dir / "f_gibbs.pth", weights_only=True)

            # Make deep copies at two particular indices
            if condition:
                f_mala = f_mala_full[-1].clone()
                f_mala_stat = f_mala_full[-7500].clone()
            f_gibbs = f_gibbs_full[-1].clone()
            f_gibbs_stat = f_gibbs_full[-7500].clone()

            # Free the allocated full tensors
            if condition:
                del f_mala_full
            del f_gibbs_full

            # Run the garbage collector
            gc.collect()

            # Basic sanity checks
            assert is_finite(samples_mala)
            assert is_finite(samples_gibbs)

            if condition:
                assert is_finite(f_mala)
            assert is_finite(f_gibbs)

            if condition:
                assert is_finite(f_mala_stat)
            assert is_finite(f_gibbs_stat)

            # Visualize empirical CDFs to see if both samplers converged to the same
            plt.figure(figsize=(4, 4))
            if condition:
                f_mala = torch.sort(f_mala)[0]
                f_mala_stat = torch.sort(f_mala_stat)[0]
            f_gibbs = torch.sort(f_gibbs)[0]
            f_gibbs_stat = torch.sort(f_gibbs_stat)[0]
            emp_cdf = torch.cumsum(
                torch.ones_like(f_mala) / f_mala.numel(), dim=0
            ).cpu()
            cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            if condition:
                plt.plot(f_mala.cpu(), emp_cdf, color=cycle[0])
                plt.plot(f_mala_stat.cpu(), emp_cdf, "--", color=cycle[0])
            plt.plot(f_gibbs.cpu(), emp_cdf, color=cycle[1])
            plt.plot(f_gibbs_stat.cpu(), emp_cdf, "--", color=cycle[1])

            plt.savefig(dir / f"{factor_name}_{patch_size}x{patch_size}_CDFs.pdf")
            plt.close()

            # Basic sanity checks
            if condition:
                assert samples_mala.shape == samples_gibbs.shape
            assert num_samples**2 <= samples_gibbs.shape[0]

            # Subselect the samples
            if condition:
                samples_mala = samples_mala[: num_samples**2]
            samples_gibbs = samples_mala[: num_samples**2]

            # Compute the value ranges for visualization purposes
            if condition:
                mala_min, mala_max = (
                    samples_mala.min().item(),
                    samples_mala.max().item(),
                )
            gibbs_min, gibbs_max = (
                samples_gibbs.min().item(),
                samples_gibbs.max().item(),
            )
            if condition:
                vmin = min(mala_min, gibbs_min)
                vmax = max(mala_max, gibbs_max)
            else:
                vmin = gibbs_min
                vmax = gibbs_max

            # Plot the samples
            for i in range(rows):
                for j in range(cols):
                    # Compute the subplot index
                    idx = i * cols + j

                    # Plot MALA samples
                    if condition:
                        plt.figure(fig_mala.number)
                        plt.subplot(rows, cols, idx + 1)
                        plt.imshow(
                            samples_mala[idx, 0, :, :].cpu(),
                            cmap="gray",
                            vmin=vmin,
                            vmax=vmax,
                        )
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()

                    # Plot Gibbs samples
                    plt.figure(fig_gibbs.number)
                    plt.subplot(rows, cols, idx + 1)
                    plt.imshow(
                        samples_gibbs[idx, 0, :, :].cpu(),
                        cmap="gray",
                        vmin=vmin,
                        vmax=vmax,
                    )
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()

            # Save figures to disk
            if condition:
                fig_mala.savefig(
                    dir / f"{factor_name}_{patch_size}x{patch_size}__mala_samples.pdf"
                )
            fig_gibbs.savefig(
                dir / f"{factor_name}_{patch_size}x{patch_size}_gibbs_samples.pdf"
            )

            # Close figures
            plt.figure(fig_mala.number)
            plt.close()

            plt.figure(fig_gibbs.number)
            plt.close()

            print("\nDone!")
    print(80 * "*")


# Computes and plots the Wasserstein distance to the marginal ground-truth distribution over the iterations
def plot_wasserstein():
    # Get the factors and patch sizes
    factor_list, patch_size_list, _ = get_configurations()

    # Get the output directory
    dir = get_output_dir()

    # Compute the Wasserstein distances
    print()
    print(80 * "*")
    print("Running Wasserstein computations")
    print(80 * "*")

    # Define the number of samples to use for visualizations
    # Gibbs
    viz_dict_12_gibbs = {
        "normal": 50,
        "laplace": 50,
        "student-t": 50,
        "gmm": 50,
    }
    viz_dict_24_gibbs = {
        "normal": 50,
        "laplace": 50,
        "student-t": 50,
        "gmm": 75,
    }
    viz_dict_48_gibbs = {
        "normal": 50,
        "laplace": 100,
        "student-t": 50,
        "gmm": 100,
    }
    viz_dict_96_gibbs = {
        "normal": 50,
        "laplace": 50,
        "student-t": 50,
        "gmm": 100,
    }
    viz_dict_gibbs = {
        12: viz_dict_12_gibbs,
        24: viz_dict_24_gibbs,
        48: viz_dict_48_gibbs,
        96: viz_dict_96_gibbs,
    }

    # MALA
    viz_dict_12_mala = {
        "normal": 1000,
        "laplace": 1000,
        "student-t": 1000,
        "gmm": 1000,
    }
    viz_dict_24_mala = {
        "normal": 1000,
        "laplace": 1000,
        "student-t": 1000,
        "gmm": 1000,
    }
    viz_dict_48_mala = {
        "normal": 5000,
        "laplace": 5000,
        "student-t": 5000,
        "gmm": 5000,
    }
    viz_dict_96_mala = {
        "normal": 5000,
        "laplace": 5000,
        "student-t": 5000,
        "gmm": 5000,
    }
    viz_dict_mala = {
        12: viz_dict_12_mala,
        24: viz_dict_24_mala,
        48: viz_dict_48_mala,
        96: viz_dict_96_mala,
    }

    # Run the experiments
    rows, cols = len(patch_size_list), len(factor_list)
    plt.figure(figsize=(4.5 * cols, 4.5 * rows))
    for i, patch_size in enumerate(patch_size_list):
        for j, factor_name in enumerate(factor_list):
            print()
            print(
                f"Computing for {factor_name} factors and patch size {patch_size} x {patch_size}..."
            )

            condition = not (
                patch_size in [48, 96] and factor_name in ["laplace", "gmm"]
            )

            # Get the dir of the current experiment
            experiment_dir = dir / factor_name / f"{patch_size}"

            # Get the number of samples
            num_samples_gibbs = viz_dict_gibbs[patch_size][factor_name] + 1
            num_samples_mala = viz_dict_mala[patch_size][factor_name] + 1

            # Load the MALA and Gibbs marginal samples
            if condition:
                f_mala_full = torch.load(
                    experiment_dir / "f_mala.pth",
                    weights_only=True,
                    map_location="cpu",
                )
                times_mala = torch.load(
                    experiment_dir / "times_mala.pth",
                    weights_only=True,
                    map_location="cpu",
                )
                times_mala[1:] = times_mala[1:] - times_mala[1]
            f_gibbs = torch.load(
                experiment_dir / "f_gibbs.pth",
                weights_only=True,
                map_location="cpu",
            )
            times_gibbs = torch.load(
                experiment_dir / "times_gibbs.pth",
                weights_only=True,
                map_location="cpu",
            )[:-1]
            times_gibbs[1:] = times_gibbs[1:] - times_gibbs[1]

            # Load the ground-truth
            f_gt = f_gibbs[-1]

            # Basic sanity checks
            # if condition:
            #   assert is_finite(f_mala_full)
            assert is_finite(f_gibbs)
            assert is_finite(f_gt)

            # Select the first few samples
            if condition:
                f_mala = f_mala_full[:num_samples_mala].clone()
                times_mala = times_mala[:num_samples_mala].clone()
            f_gibbs = f_gibbs[:num_samples_gibbs].clone()
            times_gibbs = times_gibbs[:num_samples_gibbs].clone()

            # Free the MALA allocated full tensors
            if condition:
                del f_mala_full

                # Run the garbage collector
                gc.collect()

            # Sort the samples and ground-truth
            if condition:
                f_mala = torch.sort(f_mala, dim=1)[0]
            f_gibbs = torch.sort(f_gibbs, dim=1)[0]
            f_gt = torch.sort(f_gt, dim=0)[0]

            # Compute the Wasserstein-1 distances over the iterations for MALA and Gibbs
            if condition:
                w1_mala = torch.zeros((f_mala.shape[0],), dtype=torch.float64)
                for k in range(f_mala.shape[0]):
                    w1_mala[k] = wasserstein_1_samples(f_gt, f_mala[k])

            w1_gibbs = torch.zeros((f_gibbs.shape[0],), dtype=torch.float64)
            for k in range(f_gibbs.shape[0]):
                w1_gibbs[k] = wasserstein_1_samples(f_gt, f_gibbs[k])

            # Plot the results
            idx = i * cols + j + 1
            plt.subplot(rows, cols, idx)
            cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            if condition:
                plt.plot(
                    times_mala.cpu().numpy() / 1e9, torch.log10(w1_mala), color=cycle[0]
                )
            plt.plot(
                times_gibbs.cpu().numpy() / 1e9,
                torch.log10(w1_gibbs),
                # "--",
                color=cycle[1],
            )
            if j == 0:
                plt.ylabel(f"{patch_size} x {patch_size}")
            if i == 0:
                plt.title(factor_name)
            if condition:
                xub = min(
                    times_gibbs[-1].item(),
                    times_mala[-1].item(),
                )
            else:
                xub = times_gibbs[-1].item()
            plt.xlim([0, xub / 1e9])

            # Dump the CSV files
            pd.DataFrame(
                {"x": times_mala.cpu().numpy() / 1e9, "y": torch.log10(w1_mala)}
            ).to_csv(dir / f"prior_wasserstein_mala_{i + 1}_{j + 1}.csv", index=False)

            pd.DataFrame(
                {"x": times_gibbs.cpu().numpy() / 1e9, "y": torch.log10(w1_gibbs)}
            ).to_csv(dir / f"prior_wasserstein_gibbs_{i + 1}_{j + 1}.csv", index=False)

    plt.tight_layout()
    print("\nDone!")

    plt.savefig(dir / "wasserstein.pdf")
    plt.close()
    print(80 * "*")


# Computes and plots average ACF
def plot_acf(num_lags_viz=500):
    # Basic sanity checks
    assert num_lags_viz > 0

    # Get the factors and patch sizes
    factor_list, patch_size_list, _ = get_configurations()

    # Get the output directory
    dir = get_output_dir()

    # Compute the ACFs
    print()
    print(80 * "*")
    print("Running ACF computations for marginals")
    print(80 * "*")

    # Run the experiments
    rows, cols = len(patch_size_list), len(factor_list)
    plt.figure(figsize=(4.5 * cols, 4.5 * rows))
    for i, patch_size in enumerate(patch_size_list):
        for j, factor_name in enumerate(factor_list):
            print()
            print(
                f"Computing for {factor_name} factors and patch size {patch_size} x {patch_size}..."
            )

            condition = not (
                patch_size in [48, 96] and factor_name in ["laplace", "gmm"]
            )

            # Get the dir of the current experiment
            experiment_dir = dir / factor_name / f"{patch_size}"

            # Load the MALA and Gibbs marginal samples
            if condition:
                f_mala_full = torch.load(
                    experiment_dir / "f_mala.pth",
                    weights_only=True,
                    map_location=torch.device("cpu"),
                )
            f_gibbs_full = torch.load(
                experiment_dir / "f_gibbs.pth",
                weights_only=True,
                map_location=torch.device("cpu"),
            )

            # Make deep copies at two particular indices
            if condition:
                f_mala = f_mala_full[-7500:].clone()
            f_gibbs = f_gibbs_full[-7500:].clone()

            # Free the allocated full tensors
            if condition:
                del f_mala_full
            del f_gibbs_full

            # Run the garbage collector
            gc.collect()

            # Basic sanity checks
            if condition:
                assert is_finite(f_mala)

                # Ignore chains that contain the same value over all iterations
                # This can sometimes happen in MALA and causes wrong ACF computations
                mask = np.ptp(f_mala, axis=0) != 0
                f_mala = f_mala[:, mask]

            assert is_finite(f_gibbs)

            # Compute the output shape for the ACF computations
            num_chains_mala = f_mala.shape[1]
            num_chains_gibbs = f_gibbs.shape[1]
            nlags = acf(f_gibbs[:, 0].cpu(), nlags=2000).size

            # Construct numpy arrays to store the outputs
            acf_mala = np.zeros((num_chains_mala, nlags))
            acf_gibbs = np.zeros((num_chains_gibbs, nlags))

            # Compute the ACFs
            for k in range(num_chains_mala):
                if condition:
                    acf_mala[k] = acf(f_mala[:, k].cpu(), nlags=2000)

            for k in range(num_chains_gibbs):
                acf_gibbs[k] = acf(f_gibbs[:, k].cpu(), nlags=2000)

            # Ignore the zero-lag entry
            if condition:
                acf_mala = acf_mala[:, 1:]
            acf_gibbs = acf_gibbs[:, 1:]

            # Compute the statistics
            if condition:
                acf_mala_mean = acf_mala.mean(axis=0)
                acf_mala_std = acf_mala.std(axis=0)
            acf_gibbs_mean = acf_gibbs.mean(axis=0)
            acf_gibbs_std = acf_gibbs.std(axis=0)

            # Plot the results
            idx = i * cols + j + 1
            plt.subplot(rows, cols, idx)
            cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            lags = np.array([lag + 1 for lag in range(nlags - 1)])
            if condition:
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
            if condition:
                plt.scatter(lags, acf_mala_mean, s=30, edgecolors=cycle[0])
            plt.scatter(
                lags, acf_gibbs_mean, s=30, edgecolors=cycle[0], facecolors="white"
            )
            if j == 0:
                plt.ylabel(f"{patch_size} x {patch_size}")
            if i == 0:
                plt.title(factor_name)
            plt.xlim([1, num_lags_viz])

            pd.DataFrame(
                {
                    "x": extend_np_array(lags[:500]),
                    "y": extend_np_array(acf_mala_mean[:500]),
                    "upper": extend_np_array(acf_mala_mean[:500] + acf_mala_std[:500]),
                    "lower": extend_np_array(acf_mala_mean[:500] - acf_mala_std[:500]),
                }
            ).to_csv(dir / f"acf_mala_{i + 1}_{j + 1}.csv", index=False)

            pd.DataFrame(
                {
                    "x": extend_np_array(lags[:500]),
                    "y": extend_np_array(acf_gibbs_mean[:500]),
                    "upper": extend_np_array(
                        acf_gibbs_mean[:500] + acf_gibbs_std[:500]
                    ),
                    "lower": extend_np_array(
                        acf_gibbs_mean[:500] - acf_gibbs_std[:500]
                    ),
                }
            ).to_csv(dir / f"acf_gibbs_{i + 1}_{j + 1}.csv", index=False)

            # Compute the effective sample sizes
            # Ignore all autocorrelation values below 0.05
            if condition:
                acf_mala[acf_mala < 0.05] = 0.0
            acf_gibbs[acf_gibbs < 0.05] = 0.0

            # Compute the ratio per chain
            if condition:
                ratios_mala = 1.0 / (1.0 + 2.0 * np.sum(acf_mala, axis=1))
            ratios_gibbs = 1.0 / (1.0 + 2.0 * np.sum(acf_gibbs, axis=1))

            # Report means and stds
            if condition:
                print(
                    f"Neff MALA  = {ratios_mala.mean():5.4f} +- {ratios_mala.std():.4f}"
                )
            print(
                f"Neff Gibbs = {ratios_gibbs.mean():5.4f} +- {ratios_gibbs.std():.4f}"
            )

    plt.tight_layout()
    print("\nDone!")

    plt.savefig(dir / "acf.pdf")
    plt.close()
    print(80 * "*")


########################################################################################################################
# Animation of sampling results
########################################################################################################################
# Script that creates sampling videos for all factors
def create_videos(rate, subsample=1):
    # Basic sanity checks
    assert rate > 0
    assert subsample > 0

    # Get the factors and patch sizes
    factor_list, patch_size_list = get_configurations()

    # Get the data directory
    root_dir = get_data_dir()

    # Get the output directory
    dir = root_dir / "experiments" / "sampling" / "prior"

    # Create the videos
    print()
    print(80 * "*")
    print("Creating videos for prior sampling...")
    print(80 * "*")
    patch_size_list = [12]
    for patch_size in patch_size_list:
        for factor_name in factor_list:
            print()
            print(
                f"Creating video for {factor_name} factors and patch size {patch_size} x {patch_size}..."
            )

            # Get the dir of the current experiment
            experiment_dir = dir / factor_name / f"{patch_size}"

            # Load the MALA and Gibbs marginal samples
            f_mala = torch.load(experiment_dir / "f_mala.pth", weights_only=True)[
                ::subsample
            ]
            f_gibbs = torch.load(experiment_dir / "f_gibbs.pth", weights_only=True)[
                ::subsample
            ]

            # Basic sanity checks
            assert is_finite(f_mala)
            assert is_finite(f_gibbs)

            # Compute the value ranges
            min_mala, max_mala = f_mala.min().item(), f_mala.max().item()
            min_gibbs, max_gibbs = f_gibbs.min().item(), f_gibbs.max().item()
            xmin, xmax = min(min_mala, max_mala), max(min_gibbs, max_gibbs)

            # Construct the output directory
            output_dir = (
                root_dir
                / "videos"
                / "sampling"
                / "prior"
                / factor_name
                / f"{patch_size}"
            )

            # Removing old files
            print("\nRemoving old files...")
            os.system(f"rm -rf {output_dir / '{*,.*}'}")
            print("Done!")

            # Animate all frames
            print("\nCreating video frames...")
            for i in tqdm(range(f_mala.shape[0])):
                # Get the sorted samples for the current iteration as float64
                f_mala_iter = torch.sort(f_mala[i].clone().to(torch.float64))[0]
                f_gibbs_iter = torch.sort(f_gibbs[i].clone().to(torch.float64))[0]

                # Compute the empirical densities
                plt.figure(figsize=(4, 8))
                plt.subplot(2, 1, 1)
                plt.subplots_adjust(
                    left=0.1,
                    top=0.95,
                    right=0.95,
                    bottom=0.05,
                    hspace=0.15,
                    wspace=0.15,
                )
                counts, bins = torch.histogram(f_mala_iter.cpu())
                plt.hist(bins[:-1], bins, weights=counts, density=True)
                counts, bins = torch.histogram(f_gibbs_iter.cpu())
                plt.hist(bins[:-1], bins, weights=counts, density=True)
                plt.xlim([xmin, xmax])

                # Compute the empirical CDFs
                plt.subplot(2, 1, 2)
                emp_cdf = torch.cumsum(
                    torch.ones_like(f_mala_iter) / f_mala_iter.numel(), dim=0
                ).cpu()
                plt.plot(f_mala_iter.cpu(), emp_cdf)
                plt.plot(f_gibbs_iter.cpu(), emp_cdf, "--")
                plt.xlim([xmin, xmax])

                # Save the figure
                plt.savefig(f"{output_dir}/{i:06d}.png")
                plt.close()

            # Create video
            print("Creating video...")
            os.system(
                f'ffmpeg -y -r {rate} -i "{output_dir}/%06d.png" -c:v libx264 -pix_fmt yuv420p -vf scale=-2:1080,setsar=1:1 {output_dir}/out.mp4'
            )
            print("\nDone!")
    print(80 * "*")


if __name__ == "__main__":
    # Load global plotting parameters
    init_matplotlib()

    # Compute runtimes
    compute_runtimes()

    # Plot collage
    # plot_collage(96, 8)

    # Create the sample plots
    plot_samples(8)

    # Create the Wasserstein plots
    plot_wasserstein()

    # Create the ACF plots
    plot_acf()

    # Create videos if necessary for debugging or presentation purposes
    # create_videos(rate=4, subsample=100)
