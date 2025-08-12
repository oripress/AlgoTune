import numpy as np
from typing import Any, List

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Optimized solver for the sparse PCA subproblem:
            minimize ||B - X||_F^2 + lambda * ||X||_1
            s.t. ||X_i||_2 <= 1 for each column i

        Notes on optimization:
        - Compute only top-k eigenpairs when beneficial using scipy.sparse.linalg.eigsh
          (avoids full eigh for large matrices when n_components << n).
        - Apply soft-thresholding and column-wise scaling in a fully vectorized manner
          to avoid Python loops over components.
        """
        # Parse inputs
        try:
            A = np.array(problem["covariance"], dtype=float)
            n_components = int(problem["n_components"])
            sparsity_param = float(problem["sparsity_param"])
        except Exception:
            return {"components": [], "explained_variance": []}

        # Basic validation
        if A.ndim != 2 or A.shape[0] != A.shape[1] or n_components <= 0:
            return {"components": [], "explained_variance": []}

        n = A.shape[0]
        # Prepare B (n x n_components), default zeros for missing eigenpairs
        B = np.zeros((n, n_components), dtype=float)

        # Attempt to compute only top-k eigenpairs when it is likely beneficial.
        use_eigsh = False
        eigsh_func = None
        try:
            # Lazy import to avoid hard dependency if not available
            from scipy.sparse.linalg import eigsh as _eigsh  # type: ignore

            eigsh_func = _eigsh
        except Exception:
            eigsh_func = None

        if eigsh_func is not None and n > 200 and 0 < n_components < n:
            # eigsh requires k < n
            k_try = min(n_components, n - 1)
            if k_try > 0:
                use_eigsh = True
        else:
            use_eigsh = False

        if use_eigsh:
            try:
                eigvals_k, eigvecs_k = eigsh_func(A, k=k_try, which="LA")
                # Sort descending
                idx = np.argsort(eigvals_k)[::-1]
                eigvals_k = eigvals_k[idx]
                eigvecs_k = eigvecs_k[:, idx]
                # Keep only positive eigenvalues
                pos_mask = eigvals_k > 0
                eigvals_pos = eigvals_k[pos_mask]
                eigvecs_pos = eigvecs_k[:, pos_mask]
                if eigvals_pos.size > 0:
                    kp = min(eigvals_pos.size, n_components)
                    B[:, :kp] = eigvecs_pos[:, :kp] * np.sqrt(eigvals_pos[:kp])
            except Exception:
                # Fallback to full decomposition
                use_eigsh = False

        if not use_eigsh:
            # Full eigendecomposition (safe fallback)
            try:
                eigvals, eigvecs = np.linalg.eigh(A)
            except Exception:
                return {"components": [], "explained_variance": []}

            # Keep only positive eigenvalues
            pos_mask = eigvals > 0
            eigvals_pos = eigvals[pos_mask]
            eigvecs_pos = eigvecs[:, pos_mask]

            if eigvals_pos.size > 0:
                idx = np.argsort(eigvals_pos)[::-1]
                eigvals_pos = eigvals_pos[idx]
                eigvecs_pos = eigvecs_pos[:, idx]
                kpos = min(eigvals_pos.size, n_components)
                if kpos > 0:
                    B[:, :kpos] = eigvecs_pos[:, :kpos] * np.sqrt(eigvals_pos[:kpos])

        # Vectorized soft-thresholding: prox for l1 with squared loss gives thr = lambda / 2
        thr = sparsity_param / 2.0
        if thr <= 0.0:
            X = B.copy()
        else:
            X = np.sign(B) * np.maximum(np.abs(B) - thr, 0.0)

        # Column norms and scaling: if norm > 1 -> scale to unit norm
        norms = np.linalg.norm(X, axis=0)
        mask = norms > 1.0
        if np.any(mask):
            inv = np.ones_like(norms)
            inv[mask] = 1.0 / norms[mask]
            # In-place scaling to avoid creating a new array
            X *= inv

        # Explained variance per component: x^T A x, vectorized
        AX = A.dot(X)  # shape (n, k)
        explained_variance_arr = np.sum(X * AX, axis=0)
        explained_variance: List[float] = [float(v) for v in explained_variance_arr.tolist()]

        # Final sanity checks: ensure correct shape and finite entries
        if X.shape != (n, n_components):
            # Pad/truncate columns as needed (shouldn't happen under normal flow)
            comp_fixed = np.zeros((n, n_components), dtype=float)
            take_k = min(X.shape[1], n_components)
            comp_fixed[:, :take_k] = X[:, :take_k]
            X = comp_fixed

        # Check for NaN/inf quickly
        if not np.all(np.isfinite(X)):
            return {"components": [], "explained_variance": []}

        return {"components": X.tolist(), "explained_variance": explained_variance}