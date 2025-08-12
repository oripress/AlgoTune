import numpy as np
from scipy.sparse.linalg import eigsh
class Solver:
    def solve(self, problem, **kwargs) -> dict:
        """
        Fast sparse PCA: truncated eigen-decomposition with ARPACK,
        in-place soft-threshold and projection onto ℓ2-ball.
        """
        # Parse problem
        A = np.array(problem["covariance"], dtype=np.float64)
        n = A.shape[0]
        n_comp = int(problem["n_components"])
        lam = float(problem["sparsity_param"])

        # Number of components
        k = min(n_comp, n)
        if k <= 0:
            return {
                "components": [[0.0] * n_comp for _ in range(n)],
                "explained_variance": [0.0] * n_comp
            }

        # Eigen-decomposition: dynamic choice based on k relative to n
        threshold = max(1, n // 10)
        if k <= threshold:
            # Small number of components: use ARPACK
            vals, vecs = eigsh(A, k=k, which='LA')
            # sort descending
            idx = np.argsort(vals)[::-1]
            vals = vals[idx]
            vecs = vecs[:, idx]
        else:
            # Larger number of components: full LAPACK
            vals_full, vecs_full = np.linalg.eigh(A)
            idx_full = np.argsort(vals_full)[::-1]
            vals = vals_full[idx_full][:k]
            vecs = vecs_full[:, idx_full[:k]]

        # Ensure non-negative eigenvalues, take sqrt in-place
        np.clip(vals, 0.0, None, out=vals)
        np.sqrt(vals, out=vals)
        # Scale eigenvectors: B = V * sqrt(Λ)
        Xk = vecs * vals

        # Soft-threshold (proximal step): Xk = sign(Xk) * max(|Xk| - lam/2, 0)
        thr = lam * 0.5
        signs = np.sign(Xk)
        np.abs(Xk, out=Xk)
        Xk -= thr
        np.clip(Xk, 0.0, None, out=Xk)
        Xk *= signs

        # Project columns onto ℓ2-ball of radius 1
        norms = np.linalg.norm(Xk, axis=0)
        # Avoid division by zero
        mask = norms > 1.0
        Xk[:, mask] /= norms[mask]

        # Build full component matrix (n x n_comp)
        if k < n_comp:
            X = np.zeros((n, n_comp), dtype=np.float64)
            X[:, :k] = Xk
        else:
            X = Xk

        # Explained variance: var_j = x_j^T A x_j
        M = A.dot(Xk)
        var_k = np.einsum('ij,ij->j', Xk, M)
        exp_var = [float(v) for v in var_k]
        # Pad zeros for missing components
        if k < n_comp:
            exp_var.extend([0.0] * (n_comp - k))

        return {"components": X.tolist(), "explained_variance": exp_var}