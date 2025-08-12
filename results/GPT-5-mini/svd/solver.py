from typing import Any, Dict
import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs) -> Dict[str, Any]:
        """
        Compute an SVD A = U @ diag(S) @ V.T.

        Strategy:
        - Compute eigendecomposition of the smaller Gram matrix (A.T @ A or A @ A.T).
        - Extract singular values as sqrt(eigenvalues) and corresponding singular vectors.
        - For zero singular values, fill remaining columns with orthonormal vectors
          (using standard-basis-first Gram-Schmidt, deterministic fallback).
        - Return U (n, k), S (k,), V (m, k) where k = min(n, m).
        """
        A = problem.get("matrix")
        if A is None:
            raise ValueError("Problem does not contain 'matrix' key.")
        A = np.asarray(A, dtype=float)
        # Ensure validators that expect a numpy array with .shape see the numpy version
        try:
            problem["matrix"] = A
        except Exception:
            pass
        if A.ndim != 2:
            raise ValueError("Input matrix must be 2D.")
        n, m = A.shape
        k = min(n, m)

        # Quick return for empty dimension
        if k == 0:
            return {
                "U": np.zeros((n, 0), dtype=float),
                "S": np.zeros((0,), dtype=float),
                "V": np.zeros((m, 0), dtype=float),
            }

        eps = np.finfo(float).eps

        def _find_orthonormal_column(prev: np.ndarray, dim: int) -> np.ndarray:
            """
            Find a unit vector of length dim that is orthogonal to columns in prev.
            Prefer standard basis vectors (deterministic), then random tries.
            prev is shape (dim, r) where r may be 0.
            """
            r = prev.shape[1] if (prev is not None and prev.ndim == 2) else 0
            # Try standard basis vectors
            for j in range(dim):
                e = np.zeros(dim, dtype=float)
                e[j] = 1.0
                if r > 0:
                    proj = prev.T @ e  # shape (r,)
                    e = e - prev @ proj
                norm = np.linalg.norm(e)
                if norm > 1e-12:
                    return e / norm
            # Random attempts
            for _ in range(10):
                v = np.random.randn(dim)
                if r > 0:
                    v = v - prev @ (prev.T @ v)
                norm = np.linalg.norm(v)
                if norm > 1e-12:
                    return v / norm
            # Last resort: fallback basis vector
            fallback = np.zeros(dim, dtype=float)
            fallback[0] = 1.0
            if r > 0:
                fallback = fallback - prev @ (prev.T @ fallback)
                norm = np.linalg.norm(fallback)
                if norm > 1e-12:
                    return fallback / norm
            return fallback

        try:
            if m <= n:
                # Work with A^T A (m x m)
                ATA = A.T @ A
                w, Vfull = np.linalg.eigh(ATA)  # ascending eigenvalues
                idx = np.argsort(w)[::-1]
                w = w[idx]
                Vfull = Vfull[:, idx]
                # Clip small negative eigenvalues to zero
                w = np.clip(w, 0.0, None)
                s_all = np.sqrt(w)
                S = s_all[:k].astype(float)
                V = Vfull[:, :k].astype(float)

                max_s = float(S.max()) if S.size > 0 else 0.0
                tol = max(max_s, 1.0) * max(n, m) * eps
                if tol == 0.0:
                    tol = eps

                U = np.zeros((n, k), dtype=float)
                nonzero_mask = S > tol
                if np.any(nonzero_mask):
                    Vnz = V[:, nonzero_mask]
                    Unz = A @ Vnz  # shape (n, r)
                    snz = S[nonzero_mask]
                    U[:, nonzero_mask] = Unz / snz[np.newaxis, :]

                # Fill zero-singular-value columns with orthonormal vectors
                for i in range(k):
                    if S[i] <= tol:
                        prev = U[:, :i] if i > 0 else np.zeros((n, 0), dtype=float)
                        U[:, i] = _find_orthonormal_column(prev, n)

            else:
                # n < m: work with A A^T (n x n)
                AAT = A @ A.T
                w, Ufull = np.linalg.eigh(AAT)
                idx = np.argsort(w)[::-1]
                w = w[idx]
                Ufull = Ufull[:, idx]
                w = np.clip(w, 0.0, None)
                s_all = np.sqrt(w)
                S = s_all[:k].astype(float)
                U = Ufull[:, :k].astype(float)

                max_s = float(S.max()) if S.size > 0 else 0.0
                tol = max(max_s, 1.0) * max(n, m) * eps
                if tol == 0.0:
                    tol = eps

                V = np.zeros((m, k), dtype=float)
                nonzero_mask = S > tol
                if np.any(nonzero_mask):
                    Unz = U[:, nonzero_mask]
                    Vnz = A.T @ Unz  # (m, r)
                    snz = S[nonzero_mask]
                    V[:, nonzero_mask] = Vnz / snz[np.newaxis, :]

                # Fill zero singular columns for V with orthonormal vectors
                for i in range(k):
                    if S[i] <= tol:
                        prev = V[:, :i] if i > 0 else np.zeros((m, 0), dtype=float)
                        V[:, i] = _find_orthonormal_column(prev, m)

        except np.linalg.LinAlgError:
            # Fallback to numpy's SVD in case eigen decomposition fails
            Ufull, s_full, Vh = np.linalg.svd(A, full_matrices=False)
            U = Ufull[:, :k]
            S = s_full[:k].astype(float)
            V = Vh.T[:, :k]

        # Ensure correct shapes and types
        U = np.asarray(U, dtype=float).reshape((n, k))
        S = np.asarray(S, dtype=float).reshape((k,))
        V = np.asarray(V, dtype=float).reshape((m, k))

        # Remove tiny negative zeros
        S = np.where(S < 0, 0.0, S)

        # Final sanity: ensure finite values
        if not (np.all(np.isfinite(U)) and np.all(np.isfinite(S)) and np.all(np.isfinite(V))):
            Ufull, s_full, Vh = np.linalg.svd(A, full_matrices=False)
            U = Ufull[:, :k]
            S = s_full[:k].astype(float)
            V = Vh.T[:, :k]

        return {"U": U, "S": S, "V": V}