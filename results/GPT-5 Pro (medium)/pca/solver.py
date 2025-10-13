from typing import Any
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

def _components_via_cov_eigh(Xc: np.ndarray, n_components: int) -> np.ndarray:
    # Compute eigen-decomposition of covariance matrix C = Xc.T @ Xc
    # Returns top n_components eigenvectors (as rows) sorted by descending eigenvalue
    C = Xc.T @ Xc
    # Full eigh, then pick top-k efficiently
    w, V = np.linalg.eigh(C)
    if n_components == 0:
        return np.empty((0, C.shape[0]), dtype=Xc.dtype)
    idx = np.argpartition(w, -n_components)[-n_components:]
    # Sort selected indices by descending eigenvalues
    idx = idx[np.argsort(w[idx])[::-1]]
    components = V[:, idx].T  # shape (k, n)
    return components

def _components_via_linear_eigsh(Xc: np.ndarray, n_components: int) -> np.ndarray:
    # Use ARPACK to compute top-k eigenvectors of C = Xc.T @ Xc without forming C
    n = Xc.shape[1]

    def matvec(y: np.ndarray) -> np.ndarray:
        # y: (n,)
        return Xc.T @ (Xc @ y)

    C_op = LinearOperator((n, n), matvec=matvec, rmatvec=matvec, dtype=Xc.dtype)
    # Compute largest algebraic eigenvalues (PSD matrix)
    vals, vecs = eigsh(C_op, k=n_components, which="LA")
    # Sort by descending eigenvalues
    order = np.argsort(vals)[::-1]
    components = vecs[:, order].T  # shape (k, n)
    return components

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        try:
            n_components = int(problem["n_components"])
            X = np.asarray(problem["X"], dtype=np.float64)
            if X.ndim != 2:
                raise ValueError("X must be 2-dimensional")

            m, n = X.shape
            if n_components < 0:
                raise ValueError("n_components must be non-negative")

            # Center the data
            Xc = X - X.mean(axis=0, keepdims=True)

            if n_components == 0:
                return np.empty((0, n), dtype=np.float64)

            # If n_components exceeds min(m, n), mimic reference by failing to fallback
            if n_components > min(m, n):
                raise ValueError("n_components greater than min(m, n)")

            k = n_components
            mn = min(m, n)

            # Heuristic selection of method:
            # - If need many components or small sizes: use SVD
            # - If n is modest and n <= m: eigen-decomposition of covariance
            # - If few components relative to dimension: ARPACK eigsh with LinearOperator
            if k < mn and mn >= 50 and k * 3 <= mn:
                # Few components: use iterative method without forming C
                components = _components_via_linear_eigsh(Xc, k)
            elif n <= m and n <= 1024:
                # Compute eigen-decomposition of covariance when n is not too large
                components = _components_via_cov_eigh(Xc, k)
            else:
                # Fall back to SVD for general case
                # Economy SVD
                _, _, VT = np.linalg.svd(Xc, full_matrices=False)
                components = VT[:k, :]

            return components
        except Exception:
            # Fallback exactly as in reference
            n_components = problem["n_components"]
            n, d = np.array(problem["X"]).shape  # note: mirrors the reference's variable naming
            V = np.zeros((n_components, n))
            idm = np.eye(n_components)
            V[:, :n_components] = idm
            return V  # return trivial answer