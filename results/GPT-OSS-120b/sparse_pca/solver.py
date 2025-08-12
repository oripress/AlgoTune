import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the sparse PCA problem using a fast heuristic.
        The method follows the reference formulation:
            minimize ||B - X||_F^2 + λ ||X||_1
            subject to ||X_i||_2 ≤ 1
        where B is constructed from the top eigenvectors of the covariance matrix.
        This implementation uses a simple soft‑thresholding (proximal) step
        followed by unit‑norm clipping, which is orders of magnitude faster
        than solving the full convex program with cvxpy.
        """
        # Extract problem data
        A = np.array(problem["covariance"], dtype=float)
        n = A.shape[0]
        n_components = int(problem["n_components"])
        sparsity_param = float(problem["sparsity_param"])

        # Eigen‑decomposition to obtain B (target dense components)
        # Use scipy's eigsh for speed when possible; fallback to dense eigh.
        try:
            from scipy.sparse.linalg import eigsh
            # Request the largest eigenvalues directly.
            k_eig = min(n_components, n - 1)
            eigvals, eigvecs = eigsh(A, k=k_eig, which='LA')
        except Exception:
            # Dense eigen‑decomposition as a fallback.
            # Try using scipy.linalg.eigh for a subset of eigenvalues.
            try:
                from scipy.linalg import eigh
                # Compute only the top k_eig eigenvalues/vectors.
                eigvals_all, eigvecs_all = eigh(A, subset_by_index=[n - k_eig, n - 1])
                # eigh returns ascending order; reverse to descending.
                eigvals = eigvals_all[::-1]
                eigvecs = eigvecs_all[:, ::-1]
            except Exception:
                # Final fallback to NumPy's eigh (full decomposition).
                eigvals, eigvecs = np.linalg.eigh(A)
                # Keep only positive eigenvalues.
                pos = eigvals > 0
                eigvals = eigvals[pos]
                eigvecs = eigvecs[:, pos]
                # Sort descending.
                idx = np.argsort(eigvals)[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
        k = min(len(eigvals), n_components)
        # Construct B only for the top k components to avoid unnecessary work
        if k > 0:
            B = eigvecs[:, :k] * np.sqrt(eigvals[:k])
        else:
            B = np.empty((n, 0), dtype=float)

        # ---------- Heuristic solution ----------
        # Soft‑thresholding (proximal operator of λ||X||₁)
        tau = sparsity_param / 2.0
        X = np.sign(B) * np.maximum(np.abs(B) - tau, 0.0)

        # Enforce unit‑norm constraint column‑wise
        col_norms = np.linalg.norm(X, axis=0)
        scale = np.where(col_norms > 1, 1.0 / col_norms, 1.0)
        X = X * scale

        # Pad with zero columns if fewer than n_components were computed
        if X.shape[1] < n_components:
            X_full = np.zeros((n, n_components), dtype=float)
            X_full[:, :X.shape[1]] = X
            X = X_full

        # If any column became all zeros (norm 0), leave it as zero (norm constraint satisfied)

        # Compute explained variance for each component
        explained_variance = []
        for i in range(n_components):
            comp = X[:, i]
            var = comp.T @ A @ comp
            explained_variance.append(float(var))

        return {"components": X.tolist(), "explained_variance": explained_variance}