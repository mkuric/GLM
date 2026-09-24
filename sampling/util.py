import os
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def extend_np_array(arg, n=10):
    # Basic sanity check
    assert n > 0

    return np.append(arg, [arg[-1]] * n)


# Utility function to capitalize a list
def capitalize_list(input_list):
    return [s.upper() if s == "gmm" else s.capitalize() for s in input_list]


# Utility function to capitalize a string
def capitalize(s):
    if s == "gmm":
        return s.upper()
    return s.capitalize()


# Utility function that initializes matplotlib parameters to provide better looking figures
# NOTE: The matplotlib defaults can be loaded by calling rcdefaults()
def init_matplotlib(comet=False):
    plt.rcParams.update(
        {
            "text.usetex": True,  # Use tex as interpreter for labels
            "font.weight": "bold",  # Use bold fonts
            "xtick.labelsize": 32,  # Make xtick labels larger
            "ytick.labelsize": 32,  # Make ytick labels larger
            "lines.linewidth": 5.0,  # Make lines thicker
            "axes.labelsize": 32,  # Make the labels on the axes larger
            "axes.titlesize": 32,  # Make the titles on the axes larger
            "legend.fontsize": 32,  # Make labels on the legend larger
            "savefig.dpi": 300,  # Enable to set up higher resolution output if necessary
            "figure.max_open_warning": 0,  # Get rid of matplotlib warning that too many figures are opened
        }
    )

    if comet:
        matplotlib.use("agg")


# Utility function that resets the random number generator for repeatability
def rng_seed(seed=0):
    # Set the seed everywhere
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Utility function that gets a path to the data directory where all outputs are saved
def get_data_dir():
    # Read the environment variable
    dir = os.environ.get("GLM_DATA_DIR")

    # Basic sanity check
    assert dir is not None

    # Return a path object
    return Path(dir)


def zoh(points, vals, x, num_chunks=1):
    # Basic sanity checks
    assert points.dim() == vals.dim() == x.dim() == 1
    assert points.numel() == vals.numel()
    assert num_chunks >= 1

    # Calculate chunk size based on the number of chunks
    chunk_size = (
                         x.numel() + num_chunks - 1
                 ) // num_chunks  # Ensures the last chunk may be smaller

    # Initialize the output tensor
    vals_out = torch.empty_like(x)

    # Process the chunks
    for i in range(0, x.numel(), chunk_size):
        end_i = min(i + chunk_size, x.numel())
        x_chunk = x[i:end_i]

        # Handle inner points
        left, right = points[:-1], points[1:]
        idx = (x_chunk.view(-1, 1) >= left.view(1, -1)) & (
                x_chunk.view(-1, 1) < right.view(1, -1)
        )
        idx = torch.argmax(idx.to(torch.int8), dim=1)
        vals_chunk = torch.gather(vals, dim=0, index=idx)

        # Handle left boundary
        vals_chunk[x_chunk < points[0]] = 0

        # Handle right boundary
        vals_chunk[x_chunk >= points[-1]] = vals[-1]

        # Store the result in the output tensor
        vals_out[i:end_i] = vals_chunk

    return vals_out


# Utility function that computes the Wasserstein-1 distance between two one-dimensional empirical CDFs via numerical
# integration
def wasserstein_1_distance(x, emp_cdf1, emp_cdf2):
    # Basic sanity checks
    assert x.dim() == emp_cdf1.dim() == emp_cdf2.dim() == 1
    assert x.numel() == emp_cdf1.numel() == emp_cdf2.numel()

    # Compute the Wasserstein-1 distance via numerical integration
    y = torch.abs(emp_cdf1 - emp_cdf2)[:-1]
    dx = torch.diff(x)
    return torch.sum(y * dx)


# Utility function that computes the Wasserstein-1 distance between two one-dimensional sample tensors
# NOTE: Assumes that X and Y are already sorted
def wasserstein_1_samples(X, Y):
    # Basic sanity checks
    assert X.dim() == Y.dim()
    assert X.numel() == Y.numel()

    # Compute the Wasserstein-1 distance via numerical integration
    return torch.mean(torch.abs(X - Y), dim=-1)


# Utility function that converts an implicit linear operator to a matrix
def construct_matrix_from_operator(A_op, x):
    # Extract the numbers of rows and columns of the matrix
    rows, cols = A_op(x).numel(), x.numel()

    # Compute the A matrix from the A operator
    A_matrix = torch.zeros((rows, cols), dtype=x.dtype, device=x.device)
    for i in range(cols):
        # Construct the ith unit basis vector of R^cols
        ei = torch.zeros(cols, dtype=x.dtype, device=x.device)
        ei[i] = 1.0

        # Reshape it into an (m, n) image and apply the A operator to it to obtain the ith column of the A matrix
        A_matrix[:, i] = A_op(ei.view(x.shape)).view(-1)

    return A_matrix


# Utility function that checks if all entries of a tensor are finite
def is_finite(x):
    return torch.isfinite(x).all()


if __name__ == '__main__':
    get_data_dir()
