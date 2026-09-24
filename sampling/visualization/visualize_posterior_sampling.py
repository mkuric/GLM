import imageio
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.nn import MSELoss

from sampling.util import get_data_dir, init_matplotlib, capitalize_list
from tqdm import tqdm


# Utility function that the considered models
def get_configurations():
    return ["normal", "laplace", "student-t", "gmm"]


# Utility function that defines the output directory
def get_output_dir():
    # Get the data directory
    root_dir = get_data_dir()

    return root_dir / "experiments" / "sampling" / "posterior"


# Utility function that computes peak signal-to-noise ratio for a noisy signal and its ground truth.
def psnr(signal: torch.tensor, ground_truth: torch.tensor):
    mse_loss = MSELoss()
    return -10.0 * torch.log10(mse_loss(signal, ground_truth))


def plot_denoising_results():
    # Get the considered models
    factor_list = get_configurations()

    # Get the output directory
    dir = get_output_dir()

    # Create results for denoising
    print()
    print(80 * "*")
    print(f"Creating denoising plots")
    print(80 * "*")

    # Compute the PSNR values for each test image
    input_psnr_dict = {}
    reconstruct_psnr_dict = {}
    for factor in factor_list:
        print()
        print(f"Creating denoising plots for {factor} factors...")

        # Load the ground-truth and noisy images
        experiment_dir = dir / "denoising" / factor
        ground_truth = torch.load(
            experiment_dir / "ground_truth.pth", weights_only=True
        )
        noisy_inputs = torch.load(
            experiment_dir / "noisy_inputs.pth", weights_only=True
        )

        # Compute input and reconstruction PSNR values
        input_psnr_values = torch.zeros(
            (256,), dtype=ground_truth.dtype, device=ground_truth.device
        )
        reconstruction_psnr_values = torch.zeros_like(input_psnr_values)
        for i in range(256):
            # Compute input PSNR
            input_psnr_values[i] = psnr(noisy_inputs[i], ground_truth[i])

            # Compute the posterior mean from the samples
            x_samples = torch.load(
                experiment_dir / f"{i:06}-gibbs.pth", weights_only=True
            )
            mean_img = torch.mean(x_samples, dim=0)

            # Compute reconstruction PSNR
            reconstruction_psnr_values[i] = psnr(mean_img, ground_truth[i])

        # Append to the dictionaries
        input_psnr_dict[factor] = input_psnr_values
        reconstruct_psnr_dict[factor] = reconstruction_psnr_values

        # Print values
        psnr_boost = reconstruction_psnr_values - input_psnr_values
        print(
            f"PSNR boost (min, median, mean, max):   {psnr_boost.min().item():.2f}, {psnr_boost.median().item():.2f}, {psnr_boost.mean().item():.2f}, {psnr_boost.max().item():.2f}"
        )
        print("Done!")

        # pd.DataFrame(psnr_boost.cpu().numpy()).to_csv(
        #     dir / f"{factor}.csv", index=False
        # )

    # Visualize the results
    plt.figure(figsize=(8, 8))
    plt.boxplot(
        [
            (reconstruct_psnr_dict[factor] - input_psnr_dict[factor]).cpu()
            for factor in factor_list
        ],
        tick_labels=capitalize_list(factor_list),
        boxprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2),
        medianprops=dict(linewidth=2),
        flierprops=dict(markersize=5, linewidth=2),
    )

    plt.ylabel(
        r"$\mathrm{PSNR}(\hat{x}, x_{\mathrm{gt}}) - \mathrm{PSNR}(\hat{x}_{\mathrm{naive}}, x_{\mathrm{gt}})$"
    )
    plt.ylim([-3.75, 17.15])
    plt.tight_layout()

    # Visualize some example reconstruction results
    scale = 1.35
    plt.figure(figsize=(scale * 5, scale * 8))
    for i, factor in enumerate(factor_list):
        # Select appropriate experiment directory
        experiment_dir = dir / "denoising" / factor

        #######################################################################
        # Lower end performance
        #######################################################################
        min_index = 154
        offset = 5

        if i == 0:
            # Ground-truth
            plt.subplot(8, 5, 1)
            plt.imshow(
                ground_truth[min_index].squeeze().cpu(), cmap="gray", vmin=0.0, vmax=1.0
            )
            plt.xticks([])
            plt.yticks([])
            plt.title(f"Ground-truth\n", fontsize=10)

            # Input
            plt.subplot(8, 5, 1 + offset)
            plt.imshow(
                noisy_inputs[min_index].squeeze().cpu(), cmap="gray", vmin=0.0, vmax=1.0
            )
            plt.xticks([])
            plt.yticks([])

            plt.title(
                f"{psnr(noisy_inputs[min_index], ground_truth[min_index]):.2f}",
                fontsize=10,
            )

            # Extract images for paper
            imageio.imwrite(
                dir / f"denoising_1_1.png",
                (255 * ground_truth[min_index].squeeze().cpu()).byte(),
            )

            imageio.imwrite(
                dir / f"denoising_2_1.png",
                (
                        255 * torch.clamp(noisy_inputs[min_index].squeeze().cpu(), 0.0, 1.0)
                ).byte(),
            )

        # Mean image
        x_samples = torch.load(
            experiment_dir / f"{min_index:06}-gibbs.pth", weights_only=True
        )
        min_mean_img = torch.mean(x_samples, dim=0)
        plt.subplot(8, 5, i + 2)
        plt.imshow(
            min_mean_img.squeeze().cpu(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        plt.xticks([])
        plt.yticks([])
        plt.title(
            f"{factor}\n{psnr(min_mean_img, ground_truth[min_index]):.2f}", fontsize=10
        )

        # Std image
        std_min_img = torch.std(x_samples, dim=0)
        plt.subplot(8, 5, i + 2 + offset)
        plt.imshow(std_min_img.squeeze().cpu(), cmap="gray")
        plt.xticks([])
        plt.yticks([])

        # Extract images for paper
        imageio.imwrite(
            dir / f"denoising_1_{i + 2}.png",
            (255 * torch.clamp(min_mean_img.squeeze().cpu(), 0.0, 1.0)).byte(),
        )

        vmin, vmax = std_min_img.min().item(), std_min_img.max().item()
        # print()
        # print(f"{{{100 * vmin:.3g}/{100 * vmax:.3g},")
        dump_std_img = (
            torch.round((std_min_img.squeeze() - vmin) / (vmax - vmin) * 255)
            .byte()
            .cpu()
        )
        imageio.imwrite(dir / f"denoising_2_{i + 2}.png", dump_std_img.numpy())

        #######################################################################
        # Median performance
        #######################################################################
        med_index = 46

        if i == 0:
            # Ground-truth
            plt.subplot(8, 5, 1 + 2 * offset)
            plt.imshow(
                ground_truth[med_index].squeeze().cpu(), cmap="gray", vmin=0.0, vmax=1.0
            )
            plt.xticks([])
            plt.yticks([])

            # Input
            plt.subplot(8, 5, 1 + 3 * offset)
            plt.imshow(
                noisy_inputs[med_index].squeeze().cpu(), cmap="gray", vmin=0.0, vmax=1.0
            )
            plt.xticks([])
            plt.yticks([])
            plt.title(
                f"{psnr(noisy_inputs[med_index], ground_truth[med_index]):.2f}",
                fontsize=10,
            )

            # Extract images for paper
            imageio.imwrite(
                dir / f"denoising_3_1.png",
                (255 * ground_truth[med_index].squeeze().cpu()).byte(),
            )

            imageio.imwrite(
                dir / f"denoising_4_1.png",
                (
                        255 * torch.clamp(noisy_inputs[med_index].squeeze().cpu(), 0.0, 1.0)
                ).byte(),
            )

        # Mean image
        x_samples = torch.load(
            experiment_dir / f"{med_index:06}-gibbs.pth", weights_only=True
        )
        med_mean_img = torch.mean(x_samples, dim=0)
        plt.subplot(8, 5, i + 2 + 2 * offset)
        plt.imshow(
            med_mean_img.squeeze().cpu(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{psnr(med_mean_img, ground_truth[med_index]):.2f}", fontsize=10)

        # Std image
        std_med_img = torch.std(x_samples, dim=0)
        plt.subplot(8, 5, i + 2 + 3 * offset)
        plt.imshow(std_med_img.squeeze().cpu(), cmap="gray")
        plt.xticks([])
        plt.yticks([])

        # Extract images for paper
        imageio.imwrite(
            dir / f"denoising_3_{i + 2}.png",
            (255 * torch.clamp(med_mean_img.squeeze().cpu(), 0.0, 1.0)).byte(),
        )

        vmin, vmax = std_med_img.min().item(), std_med_img.max().item()
        # print(f"{100 * vmin:.3g}/{100 * vmax:.3g},")
        dump_std_img = (
            torch.round((std_med_img.squeeze() - vmin) / (vmax - vmin) * 255)
            .byte()
            .cpu()
        )
        imageio.imwrite(dir / f"denoising_4_{i + 2}.png", dump_std_img.numpy())

        #######################################################################
        # Mean performance
        #######################################################################
        mean_index = 111

        # Ground-truth
        if i == 0:
            plt.subplot(8, 5, 1 + 4 * offset)
            plt.imshow(
                ground_truth[mean_index].squeeze().cpu(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            plt.xticks([])
            plt.yticks([])

            # Input
            plt.subplot(8, 5, 1 + 5 * offset)
            plt.imshow(
                noisy_inputs[mean_index].squeeze().cpu(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            plt.xticks([])
            plt.yticks([])
            plt.title(
                f"{psnr(noisy_inputs[mean_index], ground_truth[mean_index]):.2f}",
                fontsize=10,
            )

            # Extract images for paper
            imageio.imwrite(
                dir / f"denoising_5_1.png",
                (255 * ground_truth[mean_index].squeeze().cpu()).byte(),
            )

            imageio.imwrite(
                dir / f"denoising_6_1.png",
                (
                        255
                        * torch.clamp(noisy_inputs[mean_index].squeeze().cpu(), 0.0, 1.0)
                ).byte(),
            )

        # Mean image
        x_samples = torch.load(
            experiment_dir / f"{mean_index:06}-gibbs.pth", weights_only=True
        )
        mean_mean_img = torch.mean(x_samples, dim=0)
        plt.subplot(8, 5, i + 2 + 4 * offset)
        plt.imshow(
            mean_mean_img.squeeze().cpu(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{psnr(mean_mean_img, ground_truth[mean_index]):.2f}", fontsize=10)

        # Std image
        std_mean_img = torch.std(x_samples, dim=0)
        plt.subplot(8, 5, i + 2 + 5 * offset)
        plt.imshow(std_mean_img.squeeze().cpu(), cmap="gray")
        plt.xticks([])
        plt.yticks([])

        # Extract images for paper
        imageio.imwrite(
            dir / f"denoising_5_{i + 2}.png",
            (255 * torch.clamp(mean_mean_img.squeeze().cpu(), 0.0, 1.0)).byte(),
        )

        vmin, vmax = std_mean_img.min().item(), std_mean_img.max().item()
        # print(f"{100 * vmin:.3g}/{100 * vmax:.3g},")
        dump_std_img = (
            torch.round((std_mean_img.squeeze() - vmin) / (vmax - vmin) * 255)
            .byte()
            .cpu()
        )
        imageio.imwrite(dir / f"denoising_6_{i + 2}.png", dump_std_img.numpy())

        #######################################################################
        # Upper end performance
        #######################################################################
        max_index = 240

        # Ground-truth
        if i == 0:
            plt.subplot(8, 5, 1 + 6 * offset)
            plt.imshow(
                ground_truth[max_index].squeeze().cpu(), cmap="gray", vmin=0.0, vmax=1.0
            )
            plt.xticks([])
            plt.yticks([])

            # Input
            plt.subplot(8, 5, 1 + 7 * offset)
            plt.imshow(
                noisy_inputs[max_index].squeeze().cpu(), cmap="gray", vmin=0.0, vmax=1.0
            )
            plt.xticks([])
            plt.yticks([])
            plt.title(
                f"{psnr(noisy_inputs[max_index], ground_truth[max_index]):.2f}",
                fontsize=10,
            )

            # Extract images for paper
            imageio.imwrite(
                dir / f"denoising_7_1.png",
                (255 * ground_truth[max_index].squeeze().cpu()).byte(),
            )

            imageio.imwrite(
                dir / f"denoising_8_1.png",
                (
                        255 * torch.clamp(noisy_inputs[max_index].squeeze().cpu(), 0.0, 1.0)
                ).byte(),
            )

        # Mean image
        x_samples = torch.load(
            experiment_dir / f"{max_index:06}-gibbs.pth", weights_only=True
        )
        max_mean_img = torch.mean(x_samples, dim=0)
        plt.subplot(8, 5, i + 2 + 6 * offset)
        plt.imshow(
            max_mean_img.squeeze().cpu(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{psnr(max_mean_img, ground_truth[max_index]):.2f}", fontsize=10)

        # Std image
        std_max_img = torch.std(x_samples, dim=0)
        plt.subplot(8, 5, i + 2 + 7 * offset)
        plt.imshow(std_max_img.squeeze().cpu(), cmap="gray")
        plt.xticks([])
        plt.yticks([])

        # Extract images for paper
        imageio.imwrite(
            dir / f"denoising_7_{i + 2}.png",
            (255 * torch.clamp(max_mean_img.squeeze().cpu(), 0.0, 1.0)).byte(),
        )

        vmin, vmax = std_max_img.min().item(), std_max_img.max().item()
        # print(f"{100 * vmin:.3g}/{100 * vmax:.3g}}}")
        dump_std_img = (
            torch.round((std_max_img.squeeze() - vmin) / (vmax - vmin) * 255)
            .byte()
            .cpu()
        )
        imageio.imwrite(dir / f"denoising_8_{i + 2}.png", dump_std_img.numpy())

    plt.tight_layout()
    print(80 * "*")


def plot_dct_inpainting_results():
    # Get the considered models
    factor_list = get_configurations()

    # Get the output directory
    dir = get_output_dir()

    # Create results for DCT inpainting
    print()
    print(80 * "*")
    print(f"Creating DCT inpainting plots")
    print(80 * "*")

    # Compute the PSNR values for each test image
    input_psnr_dict = {}
    reconstruct_psnr_dict = {}
    for factor in factor_list:
        print()
        print(f"Creating DCT inpainting plots for {factor} factors...")

        # Load the ground-truth and noisy images
        experiment_dir = dir / "dct-inpainting" / factor
        ground_truth = torch.load(
            experiment_dir / "ground_truth.pth", weights_only=True
        )
        zero_fill_solutions = torch.load(
            experiment_dir / "zero_fill_solutions.pth", weights_only=True
        )

        # Compute input and reconstruction PSNR values
        input_psnr_values = torch.zeros(
            (256,), dtype=ground_truth.dtype, device=ground_truth.device
        )
        reconstruction_psnr_values = torch.zeros_like(input_psnr_values)
        for i in range(256):
            # Compute input PSNR
            input_psnr_values[i] = psnr(zero_fill_solutions[i], ground_truth[i])

            # Compute the posterior mean from the samples
            x_samples = torch.load(
                experiment_dir / f"{i:06}-gibbs.pth", weights_only=True
            )
            mean_img = torch.mean(x_samples, dim=0)

            # Compute reconstruction PSNR
            reconstruction_psnr_values[i] = psnr(mean_img, ground_truth[i])

        # Append to the dictionaries
        input_psnr_dict[factor] = input_psnr_values
        reconstruct_psnr_dict[factor] = reconstruction_psnr_values

        # Print values
        psnr_boost = reconstruction_psnr_values - input_psnr_values
        print(
            f"PSNR boost (min, median, mean, max):   {psnr_boost.min().item():.2f}, {psnr_boost.median().item():.2f}, {psnr_boost.mean().item():.2f}, {psnr_boost.max().item():.2f}"
        )
        print("Done!")

        # pd.DataFrame(psnr_boost.cpu().numpy()).to_csv(
        #     dir / f"{factor}.csv", index=False
        # )

    # Visualize the results
    plt.figure(figsize=(8, 8))
    plt.boxplot(
        [
            (reconstruct_psnr_dict[factor] - input_psnr_dict[factor]).cpu()
            for factor in factor_list
        ],
        tick_labels=capitalize_list(factor_list),
        boxprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2),
        medianprops=dict(linewidth=2),
        flierprops=dict(markersize=5, linewidth=2),
    )

    plt.ylabel(
        r"$\mathrm{PSNR}(\hat{x}, x_{\mathrm{gt}}) - \mathrm{PSNR}(\hat{x}_{\mathrm{naive}}, x_{\mathrm{gt}})$"
    )

    plt.ylim([-3.75, 17.15])
    plt.tight_layout()

    # Visualize some example reconstruction results
    scale = 1.35
    plt.figure(figsize=(scale * 5, scale * 8))
    for i, factor in enumerate(factor_list):
        # Select appropriate experiment directory
        experiment_dir = dir / "dct-inpainting" / factor

        #######################################################################
        # Lower end performance
        #######################################################################
        min_index = 154
        offset = 5

        if i == 0:
            # Ground-truth
            plt.subplot(8, 5, 1)
            plt.imshow(
                ground_truth[min_index].squeeze().cpu(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            plt.xticks([])
            plt.yticks([])
            plt.title(f"Ground-truth\n", fontsize=10)

            # Input
            plt.subplot(8, 5, 1 + offset)
            plt.imshow(
                zero_fill_solutions[min_index].squeeze().cpu(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            plt.xticks([])
            plt.yticks([])

            plt.title(
                f"{psnr(zero_fill_solutions[min_index], ground_truth[min_index]):.2f}",
                fontsize=10,
            )

            # Extract images for paper
            imageio.imwrite(
                dir / f"dct_1_1.png",
                (255 * ground_truth[min_index].squeeze().cpu()).byte(),
            )

            imageio.imwrite(
                dir / f"dct_2_1.png",
                (
                        255
                        * torch.clamp(
                    zero_fill_solutions[min_index].squeeze().cpu(), 0.0, 1.0
                )
                ).byte(),
            )

        # Mean image
        x_samples = torch.load(
            experiment_dir / f"{min_index:06}-gibbs.pth", weights_only=True
        )
        min_mean_img = torch.mean(x_samples, dim=0)
        plt.subplot(8, 5, i + 2)
        plt.imshow(
            min_mean_img.squeeze().cpu(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        plt.xticks([])
        plt.yticks([])
        plt.title(
            f"{factor}\n{psnr(min_mean_img, ground_truth[min_index]):.2f}",
            fontsize=10,
        )

        # Std image
        std_min_img = torch.std(x_samples, dim=0)
        plt.subplot(8, 5, i + 2 + offset)
        plt.imshow(std_min_img.squeeze().cpu(), cmap="gray")
        plt.xticks([])
        plt.yticks([])

        # Extract images for paper
        imageio.imwrite(
            dir / f"dct_1_{i + 2}.png",
            (255 * torch.clamp(min_mean_img.squeeze().cpu(), 0.0, 1.0)).byte(),
        )

        vmin, vmax = std_min_img.min().item(), std_min_img.max().item()
        # print()
        # print(f"{{{100 * vmin:.3g}/{100 * vmax:.3g},")
        dump_std_img = (
            torch.round((std_min_img.squeeze() - vmin) / (vmax - vmin) * 255)
            .byte()
            .cpu()
        )
        imageio.imwrite(dir / f"dct_2_{i + 2}.png", dump_std_img.numpy())

        #######################################################################
        # Median performance
        #######################################################################
        med_index = 46

        if i == 0:
            # Ground-truth
            plt.subplot(8, 5, 1 + 2 * offset)
            plt.imshow(
                ground_truth[med_index].squeeze().cpu(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            plt.xticks([])
            plt.yticks([])

            # Input
            plt.subplot(8, 5, 1 + 3 * offset)
            plt.imshow(
                zero_fill_solutions[med_index].squeeze().cpu(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            plt.xticks([])
            plt.yticks([])
            plt.title(
                f"{psnr(zero_fill_solutions[med_index], ground_truth[med_index]):.2f}",
                fontsize=10,
            )

            # Extract images for paper
            imageio.imwrite(
                dir / f"dct_3_1.png",
                (255 * ground_truth[med_index].squeeze().cpu()).byte(),
            )

            imageio.imwrite(
                dir / f"dct_4_1.png",
                (
                        255
                        * torch.clamp(
                    zero_fill_solutions[med_index].squeeze().cpu(), 0.0, 1.0
                )
                ).byte(),
            )

        # Mean image
        x_samples = torch.load(
            experiment_dir / f"{med_index:06}-gibbs.pth", weights_only=True
        )
        med_mean_img = torch.mean(x_samples, dim=0)
        plt.subplot(8, 5, i + 2 + 2 * offset)
        plt.imshow(
            med_mean_img.squeeze().cpu(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{psnr(med_mean_img, ground_truth[med_index]):.2f}", fontsize=10)

        # Std image
        std_med_img = torch.std(x_samples, dim=0)
        plt.subplot(8, 5, i + 2 + 3 * offset)
        plt.imshow(std_med_img.squeeze().cpu(), cmap="gray")
        plt.xticks([])
        plt.yticks([])

        # Extract images for paper
        imageio.imwrite(
            dir / f"dct_3_{i + 2}.png",
            (255 * torch.clamp(med_mean_img.squeeze().cpu(), 0.0, 1.0)).byte(),
        )

        vmin, vmax = std_med_img.min().item(), std_med_img.max().item()
        # print(f"{100 * vmin:.3g}/{100 * vmax:.3g},")
        dump_std_img = (
            torch.round((std_med_img.squeeze() - vmin) / (vmax - vmin) * 255)
            .byte()
            .cpu()
        )
        imageio.imwrite(dir / f"dct_4_{i + 2}.png", dump_std_img.numpy())

        #######################################################################
        # Mean performance
        #######################################################################
        mean_index = 111

        # Ground-truth
        if i == 0:
            plt.subplot(8, 5, 1 + 4 * offset)
            plt.imshow(
                ground_truth[mean_index].squeeze().cpu(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            plt.xticks([])
            plt.yticks([])

            # Input
            plt.subplot(8, 5, 1 + 5 * offset)
            plt.imshow(
                zero_fill_solutions[mean_index].squeeze().cpu(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            plt.xticks([])
            plt.yticks([])
            plt.title(
                f"{psnr(zero_fill_solutions[mean_index], ground_truth[mean_index]):.2f}",
                fontsize=10,
            )

            # Extract images for paper
            imageio.imwrite(
                dir / f"dct_5_1.png",
                (255 * ground_truth[mean_index].squeeze().cpu()).byte(),
            )

            imageio.imwrite(
                dir / f"dct_6_1.png",
                (
                        255
                        * torch.clamp(
                    zero_fill_solutions[mean_index].squeeze().cpu(), 0.0, 1.0
                )
                ).byte(),
            )

        # Mean image
        x_samples = torch.load(
            experiment_dir / f"{mean_index:06}-gibbs.pth", weights_only=True
        )
        mean_mean_img = torch.mean(x_samples, dim=0)
        plt.subplot(8, 5, i + 2 + 4 * offset)
        plt.imshow(
            mean_mean_img.squeeze().cpu(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{psnr(mean_mean_img, ground_truth[mean_index]):.2f}", fontsize=10)

        # Std image
        std_mean_img = torch.std(x_samples, dim=0)
        plt.subplot(8, 5, i + 2 + 5 * offset)
        plt.imshow(std_mean_img.squeeze().cpu(), cmap="gray")
        plt.xticks([])
        plt.yticks([])

        # Extract images for paper
        imageio.imwrite(
            dir / f"dct_5_{i + 2}.png",
            (255 * torch.clamp(mean_mean_img.squeeze().cpu(), 0.0, 1.0)).byte(),
        )

        vmin, vmax = std_mean_img.min().item(), std_mean_img.max().item()
        # print(f"{100 * vmin:.3g}/{100 * vmax:.3g},")
        dump_std_img = (
            torch.round((std_mean_img.squeeze() - vmin) / (vmax - vmin) * 255)
            .byte()
            .cpu()
        )
        imageio.imwrite(dir / f"dct_6_{i + 2}.png", dump_std_img.numpy())

        #######################################################################
        # Upper end performance
        #######################################################################
        max_index = 240

        # Ground-truth
        if i == 0:
            plt.subplot(8, 5, 1 + 6 * offset)
            plt.imshow(
                ground_truth[max_index].squeeze().cpu(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            plt.xticks([])
            plt.yticks([])

            # Input
            plt.subplot(8, 5, 1 + 7 * offset)
            plt.imshow(
                zero_fill_solutions[max_index].squeeze().cpu(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            plt.xticks([])
            plt.yticks([])
            plt.title(
                f"{psnr(zero_fill_solutions[max_index], ground_truth[max_index]):.2f}",
                fontsize=10,
            )

            # Extract images for paper
            imageio.imwrite(
                dir / f"dct_7_1.png",
                (255 * ground_truth[max_index].squeeze().cpu()).byte(),
            )

            imageio.imwrite(
                dir / f"dct_8_1.png",
                (
                        255
                        * torch.clamp(
                    zero_fill_solutions[max_index].squeeze().cpu(), 0.0, 1.0
                )
                ).byte(),
            )

        # Mean image
        x_samples = torch.load(
            experiment_dir / f"{max_index:06}-gibbs.pth", weights_only=True
        )
        max_mean_img = torch.mean(x_samples, dim=0)
        plt.subplot(8, 5, i + 2 + 6 * offset)
        plt.imshow(
            max_mean_img.squeeze().cpu(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{psnr(max_mean_img, ground_truth[max_index]):.2f}", fontsize=10)

        # Std image
        std_max_img = torch.std(x_samples, dim=0)
        plt.subplot(8, 5, i + 2 + 7 * offset)
        plt.imshow(std_max_img.squeeze().cpu(), cmap="gray")
        plt.xticks([])
        plt.yticks([])

        # Extract images for paper
        imageio.imwrite(
            dir / f"dct_7_{i + 2}.png",
            (255 * torch.clamp(max_mean_img.squeeze().cpu(), 0.0, 1.0)).byte(),
        )

        vmin, vmax = std_max_img.min().item(), std_max_img.max().item()
        # print(f"{100 * vmin:.3g}/{100 * vmax:.3g}}}")
        dump_std_img = (
            torch.round((std_max_img.squeeze() - vmin) / (vmax - vmin) * 255)
            .byte()
            .cpu()
        )
        imageio.imwrite(dir / f"dct_8_{i + 2}.png", dump_std_img.numpy())

    plt.tight_layout()
    print(80 * "*")


def interpolate(x, xp, fp):
    return torch.from_numpy(np.interp(x.cpu().numpy(), xp.cpu().numpy(), fp.cpu().numpy())).to(xp.dtype).to(xp.device)


def plot_denoising_timings():
    # Get the considered models
    factor_list = get_configurations()

    # Get the output directory
    dir = get_output_dir()

    # Create results for denoising
    print()
    print(80 * "*")
    print(f"Creating denoising timing plots")
    print(80 * "*")

    # Fix the runtime upper bounds for each test case
    t_mean_max = {"laplace": 10, "student-t": 0.2, "gmm": 2}
    t_std_max = {"laplace": 11, "student-t": 0.2, "gmm": 4}

    fig_all_runs = plt.figure(figsize=(8, 10.5))
    fig_summary = plt.figure(figsize=(8, 10.5))
    num_test_images = 256
    for factor in factor_list:
        if factor == 'normal':
            all_times_gibbs = torch.zeros(num_test_images, dtype=torch.float64)
            for idx in tqdm(range(num_test_images)):
                # Load the Gibbs marginals and timings
                experiment_dir = dir / "denoising" / factor
                times_gibbs = torch.load(experiment_dir / f"{idx:06}-times-gibbs.pth", weights_only=True)
                times_gibbs = times_gibbs - times_gibbs[0]
                times_gibbs = (times_gibbs / 1e9)

                # Store the timing
                all_times_gibbs[idx] = times_gibbs[1].clone()

            print('Normal runtimes min, max, median, mean, std [s]:', all_times_gibbs.min().item(),
                  all_times_gibbs.max().item(), all_times_gibbs.median().item(), all_times_gibbs.mean().item(),
                  all_times_gibbs.std().item())
        else:
            # Fix the common time tensor
            t_max = max(t_mean_max[factor], t_std_max[factor])
            t = torch.linspace(0, t_max, 1000, dtype=torch.float64)

            # Construct tensors to store the MSE values over the iterations at the common time grid
            mse_means_gibbs_t = torch.zeros((num_test_images, t.numel()))
            mse_stds_gibbs_t = torch.zeros((num_test_images, t.numel()))

            # Compute the MSE values over the iteration for each test image
            for idx in tqdm(range(num_test_images)):
                # Load the Gibbs marginals and timings
                experiment_dir = dir / "denoising" / factor
                marginals_gibbs = torch.load(experiment_dir / f"{idx:06}-marginals-gibbs.pth", weights_only=True)
                times_gibbs = torch.load(experiment_dir / f"{idx:06}-times-gibbs.pth", weights_only=True)
                marginals_gibbs = marginals_gibbs[:-1]
                times_gibbs = times_gibbs - times_gibbs[0]
                times_gibbs = times_gibbs[:-1]
                times_gibbs = (times_gibbs / 1e9)

                # Extract the mean and std marginals
                means_gibbs = marginals_gibbs[:, 0]
                stds_gibbs = marginals_gibbs[:, 1]

                # Compute the MSE over the iterations
                mse_loss = MSELoss()
                mse_means_gibbs = torch.zeros_like(means_gibbs[:, 0, 0, 0])
                mse_stds_gibbs = torch.zeros_like(stds_gibbs[:, 0, 0, 0])
                for i in range(marginals_gibbs.shape[0]):
                    mse_means_gibbs[i] = mse_loss(means_gibbs[i], means_gibbs[-1])
                    mse_stds_gibbs[i] = mse_loss(stds_gibbs[i], stds_gibbs[-1])

                # Interpolate the results to the common time grid
                mse_means_gibbs_t[idx] = interpolate(t, times_gibbs, mse_means_gibbs).cpu()
                mse_stds_gibbs_t[idx] = interpolate(t, times_gibbs, mse_stds_gibbs).cpu()

                # Plot the results for all runs
                plt.figure(fig_all_runs.number)
                offset = 1
                if factor == 'student-t':
                    offset = 3
                if factor == 'gmm':
                    offset = 5
                plt.subplot(3, 2, offset)
                plt.plot(times_gibbs.cpu(), mse_means_gibbs.cpu(), alpha=0.5)
                plt.plot(t, mse_means_gibbs_t[idx], '--')
                plt.xlim([t[0], t_mean_max[factor]])
                if factor == 'gmm':
                    plt.xlabel('Time [s]')
                if factor == 'laplace':
                    plt.title('Mean')
                plt.ylabel(factor)

                plt.subplot(3, 2, offset + 1)
                plt.plot(times_gibbs.cpu(), mse_stds_gibbs.cpu(), alpha=0.5)
                plt.plot(t, mse_stds_gibbs_t[idx], '--')
                plt.xlim([t[0], t_std_max[factor]])
                if factor == 'gmm':
                    plt.xlabel('Time [s]')
                if factor == 'laplace':
                    plt.title('Std')
                plt.ylabel(factor)

            # Plot the summary of the results
            plt.figure(fig_summary.number)
            cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            plt.subplot(3, 2, offset)
            plt.fill_between(
                t,
                mse_means_gibbs_t.mean(dim=0) - mse_means_gibbs_t.std(dim=0),
                mse_means_gibbs_t.mean(dim=0) + mse_means_gibbs_t.std(dim=0),
                alpha=0.5,
                color=cycle[0],
            )
            plt.plot(t, mse_means_gibbs_t.mean(dim=0), color=cycle[0])
            plt.xlim([t[0], t_mean_max[factor]])
            if factor == 'gmm':
                plt.xlabel('Time [s]')
            if factor == 'laplace':
                plt.title('Mean')
            plt.ylabel(factor)

            plt.subplot(3, 2, offset + 1)
            plt.fill_between(
                t,
                mse_stds_gibbs_t.mean(dim=0) - mse_stds_gibbs_t.std(dim=0),
                mse_stds_gibbs_t.mean(dim=0) + mse_stds_gibbs_t.std(dim=0),
                alpha=0.5,
                color=cycle[0],
            )
            plt.plot(t, mse_stds_gibbs_t.mean(dim=0))
            plt.xlim([t[0], t_std_max[factor]])
            if factor == 'gmm':
                plt.xlabel('Time [s]')
            if factor == 'laplace':
                plt.title('Std')
            plt.ylabel(factor)
            plt.figure(fig_all_runs.number)
            plt.tight_layout()
            plt.figure(fig_summary.number)
            plt.tight_layout()

            # pd.DataFrame(
            #     {
            #         "x": t.cpu().numpy(),
            #         "y": mse_means_gibbs_t.mean(dim=0).cpu().numpy(),
            #         "upper": (mse_means_gibbs_t.mean(dim=0) + mse_means_gibbs_t.std(dim=0)).cpu().numpy(),
            #         "lower": (mse_means_gibbs_t.mean(dim=0) - mse_means_gibbs_t.std(dim=0)).cpu().numpy(),
            #     }
            # ).to_csv(dir / f"denoising_{factor}_means.csv", index=False)
            #
            # pd.DataFrame(
            #     {
            #         "x": t.cpu().numpy(),
            #         "y": mse_stds_gibbs_t.mean(dim=0).cpu().numpy(),
            #         "upper": (mse_stds_gibbs_t.mean(dim=0) + mse_stds_gibbs_t.std(dim=0)).cpu().numpy(),
            #         "lower": (mse_stds_gibbs_t.mean(dim=0) - mse_stds_gibbs_t.std(dim=0)).cpu().numpy(),
            #     }
            # ).to_csv(dir / f"denoising_{factor}_stds.csv", index=False)

    print(80 * "*")


def plot_dct_inpainting_timings():
    # Get the considered models
    factor_list = get_configurations()

    # Get the output directory
    dir = get_output_dir()

    # Create results for denoising
    print()
    print(80 * "*")
    print(f"Creating dct inpainting timing plots [s]")
    print(80 * "*")

    # Fix the runtime upper bounds for each test case
    t_mean_max = {"laplace": 22.5, "student-t": 0.45, "gmm": 6}
    t_std_max = {"laplace": 25, "student-t": 0.45, "gmm": 10}

    fig_all_runs = plt.figure(figsize=(8, 10.5))
    fig_summary = plt.figure(figsize=(8, 10.5))
    num_test_images = 256
    for factor in factor_list:
        if factor == 'normal':
            all_times_gibbs = torch.zeros(num_test_images, dtype=torch.float64)
            for idx in tqdm(range(num_test_images)):
                # Load the Gibbs marginals and timings
                experiment_dir = dir / "dct-inpainting" / factor
                times_gibbs = torch.load(experiment_dir / f"{idx:06}-times-gibbs.pth", weights_only=True)
                times_gibbs = times_gibbs - times_gibbs[0]
                times_gibbs = (times_gibbs / 1e9)

                # Store the timing
                all_times_gibbs[idx] = times_gibbs[1].clone()

            print('Normal runtimes min, max, median, mean, std [s]:', all_times_gibbs.min().item(),
                  all_times_gibbs.max().item(), all_times_gibbs.median().item(), all_times_gibbs.mean().item(),
                  all_times_gibbs.std().item())
        else:
            # Fix the common time tensor
            t_max = max(t_mean_max[factor], t_std_max[factor])
            t = torch.linspace(0, t_max, 1000, dtype=torch.float64)

            # Construct tensors to store the MSE values over the iterations at the common time grid
            mse_means_gibbs_t = torch.zeros((num_test_images, t.numel()))
            mse_stds_gibbs_t = torch.zeros((num_test_images, t.numel()))

            # Compute the MSE values over the iteration for each test image
            for idx in tqdm(range(num_test_images)):
                # Load the Gibbs marginals and timings
                experiment_dir = dir / "dct-inpainting" / factor
                marginals_gibbs = torch.load(experiment_dir / f"{idx:06}-marginals-gibbs.pth", weights_only=True)
                times_gibbs = torch.load(experiment_dir / f"{idx:06}-times-gibbs.pth", weights_only=True)
                marginals_gibbs = marginals_gibbs[:-1]
                times_gibbs = times_gibbs - times_gibbs[0]
                times_gibbs = times_gibbs[:-1]
                times_gibbs = (times_gibbs / 1e9)

                # Extract the mean and std marginals
                means_gibbs = marginals_gibbs[:, 0]
                stds_gibbs = marginals_gibbs[:, 1]

                # Compute the MSE over the iterations
                mse_loss = MSELoss()
                mse_means_gibbs = torch.zeros_like(means_gibbs[:, 0, 0, 0])
                mse_stds_gibbs = torch.zeros_like(stds_gibbs[:, 0, 0, 0])
                for i in range(marginals_gibbs.shape[0]):
                    mse_means_gibbs[i] = mse_loss(means_gibbs[i], means_gibbs[-1])
                    mse_stds_gibbs[i] = mse_loss(stds_gibbs[i], stds_gibbs[-1])

                # Interpolate the results to the common time grid
                mse_means_gibbs_t[idx] = interpolate(t, times_gibbs, mse_means_gibbs).cpu()
                mse_stds_gibbs_t[idx] = interpolate(t, times_gibbs, mse_stds_gibbs).cpu()

                # Plot the results for all runs
                plt.figure(fig_all_runs.number)
                offset = 1
                if factor == 'student-t':
                    offset = 3
                if factor == 'gmm':
                    offset = 5
                plt.subplot(3, 2, offset)
                plt.plot(times_gibbs.cpu(), mse_means_gibbs.cpu(), alpha=0.5)
                plt.plot(t, mse_means_gibbs_t[idx], '--')
                plt.xlim([t[0], t_mean_max[factor]])
                if factor == 'gmm':
                    plt.xlabel('Time [s]')
                if factor == 'laplace':
                    plt.title('Mean')
                plt.ylabel(factor)

                plt.subplot(3, 2, offset + 1)
                plt.plot(times_gibbs.cpu(), mse_stds_gibbs.cpu(), alpha=0.5)
                plt.plot(t, mse_stds_gibbs_t[idx], '--')
                plt.xlim([t[0], t_std_max[factor]])
                if factor == 'gmm':
                    plt.xlabel('Time [s]')
                if factor == 'laplace':
                    plt.title('Std')
                plt.ylabel(factor)

            # Plot the summary of the results
            plt.figure(fig_summary.number)
            cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            plt.subplot(3, 2, offset)
            plt.fill_between(
                t,
                mse_means_gibbs_t.mean(dim=0) - mse_means_gibbs_t.std(dim=0),
                mse_means_gibbs_t.mean(dim=0) + mse_means_gibbs_t.std(dim=0),
                alpha=0.5,
                color=cycle[0],
            )
            plt.plot(t, mse_means_gibbs_t.mean(dim=0), color=cycle[0])
            plt.xlim([t[0], t_mean_max[factor]])
            if factor == 'gmm':
                plt.xlabel('Time [s]')
            if factor == 'laplace':
                plt.title('Mean')
            plt.ylabel(factor)

            plt.subplot(3, 2, offset + 1)
            plt.fill_between(
                t,
                mse_stds_gibbs_t.mean(dim=0) - mse_stds_gibbs_t.std(dim=0),
                mse_stds_gibbs_t.mean(dim=0) + mse_stds_gibbs_t.std(dim=0),
                alpha=0.5,
                color=cycle[0],
            )
            plt.plot(t, mse_stds_gibbs_t.mean(dim=0))
            plt.xlim([t[0], t_std_max[factor]])
            if factor == 'gmm':
                plt.xlabel('Time [s]')
            if factor == 'laplace':
                plt.title('Std')
            plt.ylabel(factor)
            plt.figure(fig_all_runs.number)
            plt.tight_layout()
            plt.figure(fig_summary.number)
            plt.tight_layout()

            # pd.DataFrame(
            #     {
            #         "x": t.cpu().numpy(),
            #         "y": mse_means_gibbs_t.mean(dim=0).cpu().numpy(),
            #         "upper": (mse_means_gibbs_t.mean(dim=0) + mse_means_gibbs_t.std(dim=0)).cpu().numpy(),
            #         "lower": (mse_means_gibbs_t.mean(dim=0) - mse_means_gibbs_t.std(dim=0)).cpu().numpy(),
            #     }
            # ).to_csv(dir / f"dct_{factor}_means.csv", index=False)
            #
            # pd.DataFrame(
            #     {
            #         "x": t.cpu().numpy(),
            #         "y": mse_stds_gibbs_t.mean(dim=0).cpu().numpy(),
            #         "upper": (mse_stds_gibbs_t.mean(dim=0) + mse_stds_gibbs_t.std(dim=0)).cpu().numpy(),
            #         "lower": (mse_stds_gibbs_t.mean(dim=0) - mse_stds_gibbs_t.std(dim=0)).cpu().numpy(),
            #     }
            # ).to_csv(dir / f"dct_{factor}_stds.csv", index=False)

    print(80 * "*")


if __name__ == "__main__":
    # Load global plotting parameters
    init_matplotlib()

    # Plot the denoising results
    plot_denoising_results()

    # Plot the dct inpainting results
    plot_dct_inpainting_results()

    # Plot the denoising timings
    plot_denoising_timings()

    # Plot the dct inpainting timings
    plot_dct_inpainting_timings()

    # Show the images
    plt.show()
