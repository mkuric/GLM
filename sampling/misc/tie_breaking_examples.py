import torch

from sampling.util import rng_seed

# Fix global datatypes and device
dtype = torch.float64
device = "cpu"


# Fixed 3d Gaussian example that tests Proposition 2.6
def fixed_3d_example():
    # Construct 3 rows that only span the first two coordinates of R^3
    K = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype, device=device
    )

    # Basic sanity check
    assert torch.linalg.matrix_rank(K).item() == 2

    # Compute the eigen decomposition of K^T K
    L, Q = torch.linalg.eigh(K.t() @ K)

    # Basic sanity check
    assert torch.allclose(K.t() @ K, Q @ torch.diag(L) @ Q.t())

    # Extract the subspace of R^3 where the distribution is proper
    L_sub, Q_sub = L[-2:], Q[:, -2:]

    # Basic sanity check
    assert torch.allclose(K.t() @ K, Q_sub @ torch.diag(L_sub) @ Q_sub.t())

    # Extract the nullspace of K
    L_null, Q_null = L[:-2], Q[:, :-2]

    # Basic sanity check
    assert torch.allclose(torch.zeros_like(K.t() @ K), Q_null @ torch.diag(L_null) @ Q_null.t())

    # Create a tie breaking term by spanning the nullspace as required in Proposition 2.2, here written out as the last
    # standard basis vector instead of Q_null
    null_tie = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype, device=device)

    # Basic sanity check
    assert torch.allclose(K.t() @ K @ null_tie.t(), torch.zeros_like(null_tie.t()))

    # Create a tie breaking term by picking arbitrary entries (this gives a tie breaking term that satisfies
    # Proposition 2.6)
    atie = torch.tensor([[1.0, 3.0, 7.0]], dtype=dtype, device=device)

    # Extend K with the two tie breaking terms
    K_null_tie = torch.vstack((K, null_tie))
    K_atie = torch.vstack((K, atie))

    # Basic sanity checks
    assert torch.linalg.matrix_rank(K_null_tie).item() == 3
    assert torch.linalg.matrix_rank(K_atie).item() == 3
    assert not torch.allclose(K_null_tie, K_atie)

    # Fix a mean vector
    mu_0 = torch.tensor([1.0, 2.0, 3.0, 0.0], dtype=dtype, device=device)

    # Compute the covariance matrices (K^T K)^-1 of the two tie-broken Gaussians
    Sigma_null_tie = torch.linalg.inv(K_null_tie.t() @ K_null_tie)
    Sigma_atie = torch.linalg.inv(K_atie.t() @ K_atie)

    # Compute the means (K^T K)^-1 K^T mu_0 of the two tie-broken Gaussians
    mu_null_tie = Sigma_null_tie @ K_null_tie.t() @ mu_0
    mu_atie = Sigma_atie @ K_atie.t() @ mu_0

    # Basic sanity checks
    assert not torch.allclose(Sigma_null_tie, Sigma_atie)
    assert not torch.allclose(mu_null_tie, mu_atie)

    # Project to the 2-dimensional subspace where the distribution is proper
    Sigma_null_tie_sub = Q_sub.t() @ Sigma_null_tie @ Q_sub
    Sigma_atie_sub = Q_sub.t() @ Sigma_atie @ Q_sub
    mu_null_tie_sub = Q_sub.t() @ mu_null_tie
    mu_atie_sub = Q_sub.t() @ mu_atie

    # Final check, on the subspace the distribution is proper, both tie breaking terms should yield the same mean and
    # covariance as per Proposition 2.6
    assert torch.allclose(Sigma_null_tie_sub, Sigma_atie_sub)
    assert torch.allclose(mu_null_tie_sub, mu_atie_sub)


# Fixed 4d Gaussian example that tests Proposition 2.6
def fixed_4d_example():
    # Construct 3 rows that only span the first two coordinates of R^4
    K = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=dtype,
        device=device,
    )

    # Basic sanity check
    assert torch.linalg.matrix_rank(K).item() == 2

    # Compute the eigen decomposition of K^T K
    L, Q = torch.linalg.eigh(K.t() @ K)

    # Basic sanity check
    assert torch.allclose(K.t() @ K, Q @ torch.diag(L) @ Q.t())

    # Extract the subspace of R^4 where the distribution is proper
    L_sub, Q_sub = L[-2:], Q[:, -2:]

    # Basic sanity check
    assert torch.allclose(K.t() @ K, Q_sub @ torch.diag(L_sub) @ Q_sub.t())

    # Extract the nullspace of K
    L_null, Q_null = L[:-2], Q[:, :-2]

    # Basic sanity check
    assert torch.allclose(torch.zeros_like(K.t() @ K), Q_null @ torch.diag(L_null) @ Q_null.t())

    # Create a tie breaking term by spanning the nullspace as required in Proposition 2.2, here written out as the last
    # two standard basis vectors instead of Q_null
    null_tie = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=dtype, device=device
    )

    # Basic sanity check
    assert torch.allclose(K.t() @ K @ null_tie.t(), torch.zeros_like(null_tie.t()))

    # Create a tie breaking term by picking arbitrary entries (this gives a tie breaking term that satisfies
    # Proposition 2.6)
    atie = torch.tensor(
        [[1.0, 3.0, 7.0, 9.0], [-1.0, 2.0, -5.0, 3.0]], dtype=dtype, device=device
    )

    # Extend K with the two tie breaking terms
    K_null_tie = torch.vstack((K, null_tie))
    K_atie = torch.vstack((K, atie))

    # Basic sanity checks
    assert torch.linalg.matrix_rank(K_null_tie).item() == 4
    assert torch.linalg.matrix_rank(K_atie).item() == 4
    assert not torch.allclose(K_null_tie, K_atie)

    # Fix a mean vector
    mu_0 = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0], dtype=dtype, device=device)

    # Compute the covariance matrices (K^T K)^-1 of the two tie-broken Gaussians
    Sigma_null_tie = torch.linalg.inv(K_null_tie.t() @ K_null_tie)
    Sigma_atie = torch.linalg.inv(K_atie.t() @ K_atie)

    # Compute the means (K^T K)^-1 K^T mu_0 of the two tie-broken Gaussians
    mu_null_tie = Sigma_null_tie @ K_null_tie.t() @ mu_0
    mu_atie = Sigma_atie @ K_atie.t() @ mu_0

    # Basic sanity checks
    assert not torch.allclose(Sigma_null_tie, Sigma_atie)
    assert not torch.allclose(mu_null_tie, mu_atie)

    # Project to the 2-dimensional subspace where the distribution is proper
    Sigma_null_tie_sub = Q_sub.t() @ Sigma_null_tie @ Q_sub
    Sigma_atie_sub = Q_sub.t() @ Sigma_atie @ Q_sub
    mu_null_tie_sub = Q_sub.t() @ mu_null_tie
    mu_atie_sub = Q_sub.t() @ mu_atie

    # Final check, on the subspace the distribution is proper, both tie breaking terms should yield the same mean and
    # covariance as per Proposition 2.6
    assert torch.allclose(Sigma_null_tie_sub, Sigma_atie_sub)
    assert torch.allclose(mu_null_tie_sub, mu_atie_sub)


# Fixed 5d Gaussian example that tests Proposition 2.6
def fixed_5d_example():
    # Construct 3 rows that only span the first two coordinates of R^5
    K = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=dtype,
        device=device,
    )

    # Basic sanity check
    assert torch.linalg.matrix_rank(K).item() == 2

    # Compute the eigen decomposition of K^T K
    L, Q = torch.linalg.eigh(K.t() @ K)

    # Basic sanity check
    assert torch.allclose(K.t() @ K, Q @ torch.diag(L) @ Q.t())

    # Extract the subspace of R^5 where the distribution is proper
    L_sub, Q_sub = L[-2:], Q[:, -2:]

    # Basic sanity check
    assert torch.allclose(K.t() @ K, Q_sub @ torch.diag(L_sub) @ Q_sub.t())

    # Extract the nullspace of K
    L_null, Q_null = L[:-2], Q[:, :-2]

    # Basic sanity check
    assert torch.allclose(torch.zeros_like(K.t() @ K), Q_null @ torch.diag(L_null) @ Q_null.t())

    # Create a tie breaking term by spanning the nullspace as required in Proposition 2.2, here written out as the last
    # three standard basis vectors instead of Q_null
    null_tie = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )

    # Basic sanity check
    assert torch.allclose(K.t() @ K @ null_tie.t(), torch.zeros_like(null_tie.t()))

    # Create a tie breaking term by picking arbitrary entries (this gives a tie breaking term that satisfies
    # Proposition 2.6)
    atie = torch.tensor(
        [
            [1.0, 3.0, 7.0, 9.0, 2.0],
            [-1.0, 2.0, -5.0, 3.0, 1.0],
            [2.0, -4.0, -5.0, 1.0, 6.0],
        ],
        dtype=dtype,
        device=device,
    )

    # Extend K with the two tie breaking terms
    K_null_tie = torch.vstack((K, null_tie))
    K_atie = torch.vstack((K, atie))

    # Basic sanity checks
    assert torch.linalg.matrix_rank(K_null_tie).item() == 5
    assert torch.linalg.matrix_rank(K_atie).item() == 5
    assert not torch.allclose(K_null_tie, K_atie)

    # Fix a mean vector
    mu_0 = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)

    # Compute the covariance matrices (K^T K)^-1 of the two tie-broken Gaussians
    Sigma_null_tie = torch.linalg.inv(K_null_tie.t() @ K_null_tie)
    Sigma_atie = torch.linalg.inv(K_atie.t() @ K_atie)

    # Compute the means (K^T K)^-1 K^T mu_0 of the two tie-broken Gaussians
    mu_null_tie = Sigma_null_tie @ K_null_tie.t() @ mu_0
    mu_atie = Sigma_atie @ K_atie.t() @ mu_0

    # Basic sanity checks
    assert not torch.allclose(Sigma_null_tie, Sigma_atie)
    assert not torch.allclose(mu_null_tie, mu_atie)

    # Project to the 2-dimensional subspace where the distribution is proper
    Sigma_null_tie_sub = Q_sub.t() @ Sigma_null_tie @ Q_sub
    Sigma_atie_sub = Q_sub.t() @ Sigma_atie @ Q_sub
    mu_null_tie_sub = Q_sub.t() @ mu_null_tie
    mu_atie_sub = Q_sub.t() @ mu_atie

    # Final check, on the subspace the distribution is proper, both tie breaking terms should yield the same mean and
    # covariance as per Proposition 2.6
    assert torch.allclose(Sigma_null_tie_sub, Sigma_atie_sub)
    assert torch.allclose(mu_null_tie_sub, mu_atie_sub)


# Randomized 12d Gaussian example that tests Proposition 2.6
def randomized_12d_example():
    # Fix random seed for repeatability
    rng_seed()

    # Construct 5 rows
    K = torch.randn((5, 12), dtype=dtype, device=device)

    # Add 3 more rows that are linearly dependent on the first five rows
    r = torch.randn((3, 5), dtype=dtype, device=device)
    K = torch.vstack((K, r @ K))

    # Basic sanity check
    # K has 8 rows, but should have only rank 5
    assert torch.linalg.matrix_rank(K).item() == 5

    # Compute the eigen decomposition of K^T K
    L, Q = torch.linalg.eigh(K.t() @ K)

    # Basic sanity check
    assert torch.allclose(K.t() @ K, Q @ torch.diag(L) @ Q.t())

    # Extract the subspace of R^12 where the distribution is proper
    L_sub, Q_sub = L[-5:], Q[:, -5:]

    # Basic sanity check
    assert torch.allclose(K.t() @ K, Q_sub @ torch.diag(L_sub) @ Q_sub.t())

    # Extract the nullspace of K
    L_null, Q_null = L[:-5], Q[:, :-5]

    # Basic sanity check
    assert torch.allclose(torch.zeros_like(K.t() @ K), Q_null @ torch.diag(L_null) @ Q_null.t())

    # Create a tie breaking term by spanning the nullspace as required in Proposition 2.2
    null_tie = Q_null.t()

    # Create a tie breaking term by randomizing a matrix (this gives almost surely a tie breaking term that satisfies
    # Proposition 2.6)
    atie = torch.randn_like(null_tie)

    # Extend K with the two tie breaking terms
    K_null_tie = torch.vstack((K, null_tie))
    K_atie = torch.vstack((K, atie))

    # Basic sanity checks
    assert torch.linalg.matrix_rank(K_null_tie).item() == 12
    assert torch.linalg.matrix_rank(K_atie).item() == 12
    assert not torch.allclose(K_null_tie, K_atie)

    # Generate random mean vector, one entry per row of the tie-broken matrices
    mu_0 = torch.randn((15,), dtype=dtype, device=device)

    # Compute the covariance matrices (K^T K)^-1 of the two tie-broken Gaussians
    Sigma_null_tie = torch.linalg.inv(K_null_tie.t() @ K_null_tie)
    Sigma_atie = torch.linalg.inv(K_atie.t() @ K_atie)

    # Compute the means (K^T K)^-1 K^T mu_0 of the two tie-broken Gaussians
    mu_null_tie = Sigma_null_tie @ K_null_tie.t() @ mu_0
    mu_atie = Sigma_atie @ K_atie.t() @ mu_0

    # Basic sanity checks
    assert not torch.allclose(Sigma_null_tie, Sigma_atie)
    assert not torch.allclose(mu_null_tie, mu_atie)

    # Project to the 5-dimensional subspace where the distribution is proper
    Sigma_null_tie_sub = Q_sub.t() @ Sigma_null_tie @ Q_sub
    Sigma_atie_sub = Q_sub.t() @ Sigma_atie @ Q_sub
    mu_null_tie_sub = Q_sub.t() @ mu_null_tie
    mu_atie_sub = Q_sub.t() @ mu_atie

    # Final check, on the subspace the distribution is proper, both tie breaking terms should yield the same mean and
    # covariance as per Proposition 2.6
    assert torch.allclose(Sigma_null_tie_sub, Sigma_atie_sub)
    assert torch.allclose(mu_null_tie_sub, mu_atie_sub)


# These examples numerically demonstrate on a few Gaussian test cases of various dimensions that Proposition 2.6 is
# correct
if __name__ == '__main__':
    # Fixed 3d example
    fixed_3d_example()

    # Fixed 4d example
    fixed_4d_example()

    # Fixed 5d example
    fixed_5d_example()

    # Randomized 12d example
    randomized_12d_example()
