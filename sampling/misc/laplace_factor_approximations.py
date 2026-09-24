import torch
from matplotlib import pyplot as plt
from torch.distributions import (
    Categorical,
    Normal,
    MixtureSameFamily,
    Exponential,
)

from sampling.util import init_matplotlib


def grad_f(x, p, distribution=False):
    # Clone the argument
    x = torch.clone(x)
    x.requires_grad = True

    # Backprop
    if distribution:
        loss = torch.sum(-p.log_prob(x))
    else:
        loss = torch.sum(p(x))
    loss.backward()

    return x.grad.detach()


# This script visualizes the two GMM parametrizations used in subsection "4.1.5. Sensitivity to GMM Parametrization" for
# the Laplace factor
def main():
    # Visualize the approximations
    dtype, device = torch.float64, "cpu"
    hr, eps, num_components_list = 0.5, 2.75e-2, [256, 512, 1024, 2048]
    x = torch.linspace(-0.5 + eps, 0.5 - eps, 10_000, dtype=dtype, device=device)

    # Fix the number of rows and columns for the plot
    num_rows, num_cols = len(num_components_list), 4
    plt.figure(figsize=(3.5 * num_cols, 3.5 * num_rows))
    for i, num_components in enumerate(num_components_list):
        # Set up means and variances
        means = torch.linspace(-hr, hr, num_components, dtype=dtype, device=device)
        vars_mean = (2 * hr / (num_components - 1)) ** 2 * torch.ones_like(means)

        # Compute weights to match a Laplace distribution
        w = torch.exp(-torch.abs(means))

        # Set up mean approximation distribution
        mixtures_mean = Categorical(w)
        components_mean = Normal(means, torch.sqrt(vars_mean))
        p_mean = MixtureSameFamily(mixtures_mean, components_mean)

        # Mean approximations
        plt.subplot(num_rows, num_cols, num_cols * i + 1)
        vals_mean = -p_mean.log_prob(x)
        vals_mean -= torch.min(vals_mean)
        vals_abs = torch.abs(x)
        plt.plot(x, vals_mean)
        plt.plot(x, vals_abs, "--")
        plt.ylabel(f"{num_components}")

        if i == 0:
            plt.title("mean apx")

        # Gradients of mean approximations
        plt.subplot(num_rows, num_cols, num_cols * i + 2)
        grad_vals_mean = grad_f(x, p_mean, True)
        grad_vals_abs = grad_f(x, torch.abs)
        plt.plot(x, grad_vals_mean)
        plt.plot(x, grad_vals_abs, "--")

        if i == 0:
            plt.title("grad mean apx")

        # Discretize the latent distribution through its inverse CDF to compute the variance of the GMM components
        latent_dist = Exponential(torch.tensor(0.5, dtype=dtype, device=device))
        ptile = 0.5e-7
        probs = torch.linspace(ptile, 1 - ptile, num_components + 1, dtype=dtype, device=device)
        z = latent_dist.icdf(probs)
        vars_var = (z[1:] + z[:-1]) / 2.0

        # Compute weights to match a Laplace distribution
        f = torch.exp(latent_dist.log_prob(z))
        w = 0.5 * (f[1:] + f[:-1]) * (z[1:] - z[:-1])

        # Set up var approximation distribution
        mixtures_var = Categorical(w)
        components_var = Normal(torch.zeros_like(vars_var), torch.sqrt(vars_var))
        p_var = MixtureSameFamily(mixtures_var, components_var)

        # Var approximations
        plt.subplot(num_rows, num_cols, num_cols * i + 3)
        vals_var = -p_var.log_prob(x)
        vals_var -= torch.min(vals_var)
        plt.plot(x, vals_var)
        plt.plot(x, vals_abs, "--")

        if i == 0:
            plt.title("var apx")

        # Gradients of var approximations
        plt.subplot(num_rows, num_cols, num_cols * i + 4)
        grad_vals_var = grad_f(x, p_var, True)
        plt.plot(x, grad_vals_var)
        plt.plot(x, grad_vals_abs, "--")

        if i == 0:
            plt.title("grad var apx")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load global plotting parameters
    init_matplotlib(comet=False)

    # Visualize the example
    main()
