import numpy as np
from tqdm import tqdm


# This function implements Algorithm 2 from the paper Stochastic Matrix-Free Equilibration of Diamond and Boyd
# (paper available at: https://link.springer.com/article/10.1007/s10957-016-0990-2)
def projected_stochastic_gradient(A, T, alpha, gamma, M):
    # Extract the input dimension
    n = A.shape[0]

    # Initialize u^0 and u_bar
    u = np.zeros(n)
    u_bar = np.zeros(n)

    # Perform the iterations
    for t in tqdm(range(1, T + 1)):
        # Construct the diagonal of D = diag(e^u_t)
        d = np.exp(u)

        # Draw s from {-1, 1}^n IID uniform
        s = np.random.choice([-1, 1], size=n)

        # Update u^t
        u_next = u - 2.0 * ((d * (A @ (d * s))) ** 2 - alpha ** 2 + gamma * u) / (
                gamma * (t + 1)
        )
        u = np.clip(u_next, -M, M)

        # Update u_bar
        u_bar = 2.0 * u / (t + 2.0) + t * u_bar / (t + 2.0)

    # Return the diagonal of the computed preconditioner
    return np.exp(u_bar)


# This example shows that the stochastic matrix free equilibration algorithm performs similarly to vanilla diagonal
# preconditioning on a toy example
def main():
    # Fix the random seed
    np.random.seed(0)

    # Generate a random matrix
    n = 5
    A = np.random.rand(n, n)

    # Make it symmetric
    A = (A + A.T) / 2

    # Compute its Eigendecomposition
    _, eigvecs = np.linalg.eigh(A)

    # Construct a well conditioned symmetric core
    eigvals = np.linspace(1.0, 2.0, n)
    B = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Destroy the scaling of the rows and columns
    scales = np.logspace(-2, 2, n)
    A = np.diag(scales) @ B @ np.diag(scales)

    # Symmetrize to compensate potential round-off errors
    A = (A + A.T) / 2

    # Compute the plain diagonal preconditioner
    diag_A = np.diag(A)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(diag_A))
    A_precond = D_inv_sqrt @ A @ D_inv_sqrt

    # Compute the stochastic diagonal preconditioner
    d_stochastic = projected_stochastic_gradient(A, T=10_000, alpha=1.0, gamma=0.01, M=np.log(1e4))
    A_stochastic_precond = d_stochastic[:, None] * A * d_stochastic[None, :]

    # Compare the condition numbers
    print("Condition number of A        :        ", np.linalg.cond(A))
    print("Condition number of A precond:        ", np.linalg.cond(A_precond))
    print("Condition number of A stoch. precond: ", np.linalg.cond(A_stochastic_precond))

    # Col and row norms
    print()
    print("Col and row norms of A:")
    print(np.linalg.norm(A, axis=0))
    print(np.linalg.norm(A, axis=1))

    print()
    print("Col and row norms of A precond:")
    print(np.linalg.norm(A_precond, axis=0))
    print(np.linalg.norm(A_precond, axis=1))

    print()
    print("Col and row norms of A stoch. precond:")
    print(np.linalg.norm(A_stochastic_precond, axis=0))
    print(np.linalg.norm(A_stochastic_precond, axis=1))


if __name__ == '__main__':
    main()
