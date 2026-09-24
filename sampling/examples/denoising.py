from functools import partial

import torch
from matplotlib import pyplot as plt

from sampling.samplers import (
    gibbs_sampler,
    construct_adj_conv,
    mu_map_gmm,
    var_map_gmm,
    gmm_lat_sampler,
)
from sampling.util import rng_seed

# Fix global datatype and device
dtype = torch.float64
device = "cuda"


# Utility function that computes the peak signal-to-noise ratio of an image in [0, 1] with respect to the ground truth
def psnr(x, x_gt):
    return 10.0 * torch.log10(1.0 / torch.mean((x - x_gt) ** 2)).item()


# Constructs the GMM prior used in the paper
def construct_gmm_prior():
    # Filters
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
    conv.weight.data[:, :, :, :] = 0.0

    # Horizontal
    conv.weight.data[0, 0, 1, 1] = -1.0
    conv.weight.data[0, 0, 1, 2] = 1.0

    # Vertical
    conv.weight.data[1, 0, 1, 1] = -1.0
    conv.weight.data[1, 0, 2, 1] = 1.0

    # Adjoint filters
    conv.to(device)
    adj_conv = construct_adj_conv(conv)

    # Scaling of the filter responses to match the value range of the GMM
    scaling = 0.5803

    # GMM parameters, one row per filter
    weights = torch.tensor(
        2 * [[0.1058, 0.0054, 0.0886, 0.0733, 0.0667, 0.0077, 0.0591, 0.3213, 0.2721]], dtype=dtype, device=device
    )
    means = torch.zeros_like(weights)
    variances = torch.tensor(
        2 * [[2.5e-05, 4.0e-04, 6.25e-04, 4.9e-03, 5.625e-03, 7.29e-02, 7.5625e-02, 7.84e-02, 8.1225e-02]],
        dtype=dtype,
        device=device,
    )

    # Latent maps and latent sampler of the GMM
    mu_map = partial(mu_map_gmm, means)
    var_map = partial(var_map_gmm, variances)
    latent_sampler = partial(gmm_lat_sampler, weights, means, torch.sqrt(variances))

    return conv, adj_conv, scaling, mu_map, var_map, latent_sampler


# Minimal example that samples the posterior of a denoising problem with the GMM prior from the paper
def main(num_chains=64, num_iters=500, burn_in=100, std=0.1):
    # Fix random seed for repeatability
    rng_seed()

    # Load a test image from BSD500 and simulate a noisy measurement
    x_gt = torch.load("samples_bsd500.pth", weights_only=True)[111].to(dtype=dtype, device=device)
    b = x_gt + std * torch.randn_like(x_gt)

    # Construct the prior
    conv, adj_conv, scaling, mu_map, var_map, latent_sampler = construct_gmm_prior()

    # Construct the linear operator of the model and its adjoint. The first channel is the forward operator of the
    # inverse problem (the identity for denoising), the remaining channels are the filters of the prior.
    def K(x):
        with torch.no_grad():
            return torch.concat((x, conv(scaling * x)), dim=1)

    def adj_K(y):
        with torch.no_grad():
            return y[:, :1, :, :] + adj_conv(scaling * y[:, 1:, :, :])

    # Construct the latent maps of the posterior. The first channel is the Gaussian likelihood with mean b and
    # variance std^2, the remaining channels are the GMM prior
    def mu_map_posterior(z):
        mu = torch.zeros(z.shape, dtype=dtype, device=device)
        mu[:, 0, :, :] = b
        mu[:, 1:, :, :] = mu_map(z[:, 1:, :, :])
        return mu

    def var_map_posterior(z):
        var = torch.zeros(z.shape, dtype=dtype, device=device)
        var[:, 0, :, :] = std ** 2
        var[:, 1:, :, :] = var_map(z[:, 1:, :, :])
        return var

    # The latent variables of the GMM are the indices of the mixture components, the likelihood has no latent variable
    def latent_sampler_posterior(u):
        z = torch.ones(u.shape, dtype=torch.int64, device=device)
        z[:, 1:, :, :] = latent_sampler(u[:, 1:, :, :])
        return z

    # Accumulate the first two moments of the samples after the burn-in phase to estimate the posterior mean and the
    # posterior standard deviation
    moments = {"sum": torch.zeros_like(b), "sum_sqr": torch.zeros_like(b), "count": 0}

    def accumulate(idx, x):
        if idx >= burn_in:
            moments["sum"] += x.sum(dim=0)
            moments["sum_sqr"] += (x ** 2).sum(dim=0)
            moments["count"] += x.shape[0]

    # Run the Gibbs sampler, all chains are initialized with zeros
    x_init = torch.zeros((num_chains, *b.shape), dtype=dtype, device=device)
    samples = gibbs_sampler(K, adj_K, mu_map_posterior, var_map_posterior, latent_sampler_posterior, x_init, num_iters,
                            callback=accumulate)

    # Compute the posterior mean and standard deviation
    x_mmse = moments["sum"] / moments["count"]
    x_std = torch.sqrt(moments["sum_sqr"] / moments["count"] - x_mmse ** 2)

    # Print the PSNR values
    print(f"PSNR of the noisy image:    {psnr(b, x_gt):.2f} dB")
    print(f"PSNR of the posterior mean: {psnr(x_mmse, x_gt):.2f} dB")

    # Visualize the results
    images = [
        ("Ground truth", x_gt, (0.0, 1.0)),
        (f"Noisy ({psnr(b, x_gt):.2f} dB)", b, (0.0, 1.0)),
        (f"Posterior mean ({psnr(x_mmse, x_gt):.2f} dB)", x_mmse, (0.0, 1.0)),
        ("Posterior standard deviation", x_std, None),
        ("Posterior sample", samples[0], (0.0, 1.0)),
    ]
    plt.figure(figsize=(4 * len(images), 4.5))
    for i, (title, image, value_range) in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        vmin, vmax = value_range if value_range is not None else (None, None)
        plt.imshow(image[0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
