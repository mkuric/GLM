import math
import time
from functools import partial

import logsumexpv2 as lse
import torch
from torch.distributions import (
    Gamma,
    Normal,
    Laplace,
    StudentT,
    Categorical,
    MixtureSameFamily,
)
from tqdm import tqdm


########################################################################################################################
# Utility functions
########################################################################################################################
# Utility callback function that collects the samples
def timed_collector(times, collection, idx, item, transform=lambda x: x):
    times[idx - 1] = time.perf_counter_ns()
    collection[idx] = transform(item)


def collector(collection, idx, item, transform=lambda x: x):
    collection[idx] = transform(item)


# Utility function that computes the potential of a torch distribution
def potential(distribution, x):
    return -distribution.log_prob(x)


# Constructs a torch distribution from dictionary
def construct_distribution(factor_name, value_dict, dtype, device):
    # Basic sanity check
    assert factor_name in ["normal", "laplace", "student-t", "gmm"]

    if factor_name == "normal":
        mean = torch.tensor(value_dict["mean"], dtype=dtype, device=device)
        var = torch.tensor(value_dict["var"], dtype=dtype, device=device)
        return Normal(mean, torch.sqrt(var))

    if factor_name == "laplace":
        loc = torch.tensor(0.0, dtype=dtype, device=device)
        b = torch.tensor(value_dict["b"], dtype=dtype, device=device)
        return Laplace(loc, b)

    if factor_name == "student-t":
        df = torch.tensor(value_dict["df"], dtype=dtype, device=device)
        return StudentT(df)

    # GMM as default
    weights = torch.tensor(value_dict["weights"], dtype=dtype, device=device)
    means = torch.tensor(value_dict["means"], dtype=dtype, device=device)
    vars = torch.tensor(value_dict["vars"], dtype=dtype, device=device)

    mixtures = Categorical(weights)
    components = Normal(means, torch.sqrt(vars))
    return MixtureSameFamily(mixtures, components)


# Constructs a torch distribution from dictionary
def construct_gibbs_config(factor_name, value_dict, dtype, device):
    # Basic sanity check
    assert factor_name in ["normal", "laplace", "student-t", "gmm"]

    if factor_name == "normal":
        mean = torch.tensor(value_dict["mean"], dtype=dtype, device=device)
        var = torch.tensor(value_dict["var"], dtype=dtype, device=device)
        return (
            partial(mu_map_normal, mean),
            partial(var_map_normal, var),
            normal_latent_sampler,
        )

    if factor_name == "laplace":
        b = torch.tensor(value_dict["b"], dtype=dtype, device=device)
        return zero_map, identity_map, partial(laplace_latent_sampler, b.item())

    if factor_name == "student-t":
        df = torch.tensor(value_dict["df"], dtype=dtype, device=device)
        return (zero_map, reciprocal_map, partial(student_t_latent_sampler, df))

    # GMM as default
    weights = torch.tensor(value_dict["weights"], dtype=dtype, device=device)
    means = torch.tensor(value_dict["means"], dtype=dtype, device=device)
    vars = torch.tensor(value_dict["vars"], dtype=dtype, device=device)

    mu_map = partial(mu_map_gmm, means)
    var_map = partial(var_map_gmm, vars)
    latent_sampler = partial(gmm_lat_sampler, weights, means, torch.sqrt(vars))

    return (mu_map, var_map, latent_sampler)


########################################################################################################################
# MALA sampler
########################################################################################################################
# Utility function that computes the log transition probability from x to x_prime in MALA
def log_q(x_prime, x, tau, grad_log_pi):
    batch_size = x.shape[0]
    y = (x_prime - x - tau * grad_log_pi(x)).view(batch_size, -1)
    return -(torch.sum(y**2, dim=1)) / (4.0 * tau)


# Metropolis Adjusted Langevin Algorithm (MALA) sampler
def mala_sampler(
    log_p,
    grad_log_p,
    tau,
    x_init,
    num_iters,
    verbose=True,
    callback=lambda idx, x: None,
):
    # Basic sanity check
    assert num_iters > 0
    assert tau > 0.0

    # Extract the number of samplers
    num_samplers = x_init.shape[0]

    # Draw samples
    x_sample = x_init.clone().detach()
    accepted = 0

    pb = tqdm(range(num_iters)) if verbose else range(num_iters)
    for i in pb:
        # Generate proposal states
        x_prop = (
            x_sample
            + tau * grad_log_p(x_sample)
            + math.sqrt(2.0 * tau) * torch.randn_like(x_init)
        )

        # Compute the acceptance threshold
        z1 = log_p(x_prop) - log_p(x_sample)
        z2 = log_q(x_sample, x_prop, tau, grad_log_p) - log_q(
            x_prop, x_sample, tau, grad_log_p
        )
        alpha = torch.exp(z1 + z2)
        alpha = torch.minimum(alpha, torch.tensor(1.0, dtype=x_init.dtype))

        # Accept or reject the state proposals
        select = (
            torch.rand((num_samplers,), dtype=x_init.dtype, device=x_init.device)
            <= alpha
        )
        x_sample[select] = x_prop[select]
        accepted += torch.sum(select).item()

        # Update the acceptance rate in the progress bar
        if verbose:
            pb.set_postfix_str(
                f"acceptance rate: {accepted / (num_samplers * (i + 1)):.3f}"
            )

        # Do some bookkeeping if specified
        callback(i, x_sample)

    return x_sample


########################################################################################################################
# Gibbs sampler
########################################################################################################################
# Gibbs sampler for Gaussian latent machines
def gibbs_sampler(
    K,
    adj_K,
    mu_map,
    var_map,
    latent_sampler,
    x_init,
    num_iters,
    tie_break=False,
    callback=lambda idx, x: None,
    adj_K_sqr=None,
    verbose=True,
):
    # Draw samples
    x_sample = x_init.clone().detach()
    iterable = tqdm(range(num_iters)) if verbose else range(num_iters)
    for i in iterable:
        # Generate latent sample
        z = latent_sampler(K(x_sample))

        # Map to mu and var
        mu = mu_map(z)
        var = var_map(z)

        # Perturb and map
        y = mu + torch.sqrt(var) * torch.randn_like(mu)

        # Don't apply perturb and map to the masked out elements
        y[~torch.isfinite(var)] = 0.0

        # This is the scalar case in which perturb and map reduces to direct sampling
        if mu.dim() == 2 and mu.shape[1] == 1:
            x_sample = y
        else:
            # Construct A and b
            # NOTE: This provides zero mean tie-breaking if specified
            # NOTE: When using `cg_`, we need to index the variances at the indices
            # specified in the second argument to A.
            # Same goes for M
            # We need to see for which experiments this is worth it, if at all
            def A(x):
                res = adj_K(K(x) / var)

                if tie_break:
                    res += torch.mean(x, dim=(2, 3), keepdim=True)

                return res

            b = adj_K(y / var)

            # Generate the samples
            dims = (1,) if mu.dim() == 2 else (1, 2, 3)

            if adj_K_sqr is None:
                x_sample = cg(A, b, x_sample, dims=dims)
            else:
                M = adj_K_sqr(1.0 / var)
                if tie_break:
                    num_pixels = x_sample.shape[-1] * x_sample.shape[-2]
                    M += 1.0 / num_pixels

                def M_inv(x):
                    return x / M

                def Ai(x, i):
                    res = adj_K(K(x) / var[i])

                    if tie_break:
                        res += torch.mean(x, dim=(2, 3), keepdim=True)

                    return res

                def M_invi(x, i):
                    return x / M[i]

                # x_sample = precond_cg(M_inv, A, b, x_sample, dims=dims)
                x_sample = precond_cg_(M_invi, Ai, b, x_sample, dims=dims)

        # Do some bookkeeping if specified
        callback(i, x_sample)

    return x_sample


# batched preconditioned_cg_ (with early exit) is to precond_cg what cg_ is to cg
def precond_cg_(M_inv, A, b, x0, tol=1e-4, dims=(1, 2, 3)):
    # Basic sanity checks
    assert tol > 0.0
    assert len(x0.shape) > 1

    to_do = torch.arange(b.shape[0], device=b.device)
    x = x0.clone()
    r = b - A(x, to_do)
    z = M_inv(r, to_do).clone()
    p = z.clone()

    def ip(a, b):
        return torch.sum(a * b, dim=dims, keepdim=True)

    # Perform the iterations
    while len(to_do):
        Ap = A(p, to_do).clone()
        alpha = ip(r, z) / ip(p, Ap)
        x[to_do] += alpha * p
        r_prev = torch.clone(r)
        r -= alpha * Ap
        cond = (r**2).sum(dim=dims).sqrt() > tol
        r, to_do = r[cond], to_do[cond]
        z_prev = torch.clone(z)
        z = M_inv(r, to_do).clone()
        beta = ip(r, z) / ip(r_prev, z_prev)[cond]
        p = z + beta * p[cond]

    return x


# Batch preconditioned conjugate-gradient method for solving the system Ax = b with preconditioner M.
# This is equivalent to solving the system B^{-T} A B^{-1} y = B^{-T} where M = B^T B and B^{-1} y = x.
#
# Note that the preconditioned iterations are carried out on x and the true residual (and thus the tol argument refers
# to the norm of the residual of the original system Ax = b).
#
# See the following slides for more details:
# http://www.seas.ucla.edu/~vandenbe/236C/lectures/cg.pdf
#
def precond_cg(M_inv, A, b, x0, tol=1e-4, dims=(1, 2, 3)):
    # Basic sanity checks
    assert tol > 0.0
    assert len(x0.shape) > 1

    # Compute the residual vector
    # NOTE: Deep copy to not modify the input tensor
    x = x0.clone()
    r = b - A(x)

    # NOTE: We need a deep copy to protect from trivial identity operators
    z = M_inv(r).clone()

    # Compute the next search direction
    # NOTE: We need a deep copy, otherwise p and z point to the same tensor
    p = z.clone()

    def ip(a, b):
        return torch.sum(a * b, dim=dims, keepdim=True)

    # Perform the iterations
    while torch.max((r**2).sum(dim=dims).sqrt()) > tol:
        # Pick the step-size
        # NOTE: We need a deep copy to protect from trivial identity operators
        Ap = A(p).clone()
        alpha = ip(r, z) / ip(p, Ap)

        # NOTE: This is required in case one of the systems in batch mode is already solved, which causes that
        # particular solution to become nan in the next iteration.
        alpha[~torch.isfinite(alpha)] = 0.0

        # Update the solution
        x += alpha * p

        # Update the residual vector
        # NOTE: We need a deep copy, otherwise r_prev and r point to the same tensor
        r_prev = torch.clone(r)
        r -= alpha * Ap

        # Update z
        # NOTE: We need a deep copy, otherwise z_prev and z point to the same tensor
        z_prev = torch.clone(z)

        # NOTE: We need a deep copy to protect from trivial identity operators
        z = M_inv(r).clone()

        # Update the search direction
        beta = ip(r, z) / ip(r_prev, z_prev)

        # NOTE: This is required in case one of the systems in batch mode is already solved, which causes that
        # particular solution to become nan in the next iteration.
        beta[~torch.isfinite(beta)] = 0.0

        p = z + beta * p

    return x


# Batch conjugate-gradient method for solving the system Ax = b, where A is a real, symmetric, and
# positive-definite matrix
def cg_(A, b, x0, tol=1e-4, dims=(1, 2, 3)):
    # Basic sanity checks
    assert tol > 0.0
    assert len(x0.shape) > 1

    to_do = torch.arange(b.shape[0], device=b.device)
    x = x0.clone()
    r = b - A(x, to_do)
    p = r.clone()

    def ip(a, b):
        return torch.sum(a * b, dim=dims, keepdim=True)

    # Perform the iterations
    while len(to_do):
        Ap = A(p, to_do).clone()
        alpha = ip(r, r) / ip(p, Ap)
        x[to_do] += alpha * p
        r_prev = torch.clone(r)
        r -= alpha * Ap
        cond = (r**2).sum(dim=dims).sqrt() > tol
        r, to_do = r[cond], to_do[cond]
        beta = ip(r, r) / ip(r_prev, r_prev)[cond]
        p = r + beta * p[cond]

    return x


def cg(A, b, x0, tol=1e-4, dims=(1, 2, 3)):
    # Basic sanity checks
    assert tol > 0.0
    assert len(x0.shape) > 1

    # Compute the residual vector
    # NOTE: Deep copy to not modify the input tensor
    x = x0.clone()
    r = b - A(x)

    # Compute the next search direction
    # NOTE: We need a deep copy, otherwise r and p point to the same tensor and make the iterations wrong
    p = r.clone()

    def ip(a, b):
        return torch.sum(a * b, dim=dims, keepdim=True)

    # Perform the iterations
    while torch.max((r**2).sum(dim=dims).sqrt()) > tol:
        # Pick the step-size
        # NOTE: We need a deep copy to protect from trivial identity operators
        Ap = A(p).clone()
        alpha = ip(r, r) / ip(p, Ap)

        # NOTE: This is required in case one of the systems in batch mode is already solved, which causes that
        # particular solution to become nan in the next iteration.
        alpha[~torch.isfinite(alpha)] = 0.0

        # Update the solution
        x += alpha * p

        # Update the residual vector
        # NOTE: We need a deep copy, otherwise r_prev and r point to the same tensor and beta gets fixed to one
        r_prev = torch.clone(r)
        r -= alpha * Ap

        # Update the search direction
        beta = ip(r, r) / ip(r_prev, r_prev)

        # NOTE: This is required in case one of the systems in batch mode is already solved, which causes that
        # particular solution to become nan in the next iteration.
        beta[~torch.isfinite(beta)] = 0.0

        p = r + beta * p

    return x


########################################################################################################################
# Gibbs sampler utility functions specific to the considered potentials
########################################################################################################################
# Mean and var mappings
# Utility function that maps z to a zero-vector of the same size
def zero_map(z):
    return torch.zeros_like(z)


# Utility function that maps z to itself
def identity_map(z):
    return z


# Utility function that maps z to a vector of the same size where the ith entry is defined as 1 / zi
def reciprocal_map(z):
    return 1.0 / z


# Utility function that defines the mean map for a normal distribution
def mu_map_normal(mean, z):
    return mean * z


# Utility function that defines the var map for a normal distribution
def var_map_normal(var, z):
    return var * z


# Utility function that defines the mean map for a GMM distribution
def mu_map_gmm(means, z):
    if z.dim() == 2:
        z = z[:, None, :, None]
        dims = z.shape
        return torch.gather(
            means[None, :, None, :].expand(dims[0], -1, dims[2], -1), dim=-1, index=z
        )[:, 0, :, 0]

    z = z.unsqueeze(-1)
    dims = z.shape
    return torch.gather(
        means[None, :, None, None, :].expand(dims[0], -1, dims[2], dims[3], -1),
        dim=-1,
        index=z,
    ).squeeze()


# Utility function that defines the var map for a GMM distribution
def var_map_gmm(variances, z):
    if z.dim() == 2:
        z = z[:, None, :, None]
        dims = z.shape
        return torch.gather(
            variances[None, :, None, :].expand(dims[0], -1, dims[2], -1),
            dim=-1,
            index=z,
        )[:, 0, :, 0]

    z = z.unsqueeze(-1)
    dims = z.shape
    return torch.gather(
        variances[None, :, None, None, :].expand(dims[0], -1, dims[2], dims[3], -1),
        dim=-1,
        index=z,
    ).squeeze()


# Conditional latent samplers
# Utility function that defines the conditional latent sampler of the Normal distribution
def normal_latent_sampler(u):
    return torch.ones_like(u)


# Utility function that defines the conditional latent sampler of the Laplace distribution
def laplace_latent_sampler(b, u):
    # Basic sanity check
    assert b > 0.0

    # Compute the GIG parameters for the latent samplers
    a_gig = 1.0 / (b**2) * torch.ones_like(u)
    b_gig = u**2
    p_gig = 0.5 * torch.ones_like(u)

    # NOTE: We get a division by zero error in the gig sampler if one of the b parameters is zero (this is equivalent to
    # one of the u components being zero)
    b_gig = torch.maximum(b_gig, 1e-7 * torch.ones_like(b_gig))

    # Draw the samples
    return lse.gig_sampler(a_gig, b_gig, p_gig)


# Utility function that defines the conditional latent sampler of the student-t distribution
def student_t_latent_sampler(df, u):
    # Basic sanity check
    assert df > 0.0

    cond_latent_dist = Gamma((df + 1.0) / 2.0, (df + (u**2)) / 2)
    return cond_latent_dist.sample()


# Utility function that defines the conditional latent sampler of the GMM distribution
def gmm_lat_sampler(weights, means, sigmas, u):
    if u.dim() == 2:
        return lse.categorical_sampler(u[:, None, None, :], weights, means, sigmas)[
            :, 0, 0, :
        ]

    return lse.categorical_sampler(u, weights, means, sigmas)


# Utility function that constructs the adjoint convolution for a convolutional layer with circular boundary conditions
def construct_adj_conv(conv):
    # Construct the adjoint operator as a convolutional layer
    adj_conv = torch.nn.Conv2d(
        in_channels=conv.out_channels,
        out_channels=conv.in_channels,
        kernel_size=conv.kernel_size,
        bias=False,
        dtype=conv.weight.dtype,
        device=conv.weight.device,
        padding_mode=conv.padding_mode,
        padding=conv.padding,
    )
    adj_conv.weight.requires_grad = conv.weight.requires_grad

    # Construct the weights from the argument convolutional layer
    for i in range(conv.out_channels):
        kernel = conv.weight.data[i, 0, :, :]
        adj_conv.weight.data[0, i, :, :] = torch.rot90(torch.rot90(kernel))

    return adj_conv
