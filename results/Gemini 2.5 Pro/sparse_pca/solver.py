import numpy as np
from typing import Any, Dict
import numba
from scipy.sparse.linalg import eigsh

# Numba-jitted FISTA solver for the subproblem.
# Using cache=True to avoid re-compilation on subsequent runs.
@numba.jit(nopython=True, fastmath=True, cache=True)
def _solve_component_fista(b_i: np.ndarray, sparsity_param: float, t: float, max_iter: int, tol: float) -> np.ndarray:
    """
    Solves the subproblem for a single component using FISTA.
    min ||b - x||^2 + sparsity_param * ||x||_1 s.t. ||x||_2 <= 1
    """
    # Initialization for FISTA
    x_i = b_i.copy()
    y_i = x_i.copy()
    s = 1.0

    for _ in range(max_iter):
        x_old = x_i.copy()

        # Gradient step on y_i
        grad = 2.0 * (y_i - b_i)
        z = y_i - t * grad

        # Proximal operator for L1 norm (soft-thresholding)
        threshold = t * sparsity_param
        z_thresh = np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)

        # Projection onto the L2 unit ball
        norm_z = np.linalg.norm(z_thresh)
        if norm_z > 1.0:
            x_i = z_thresh / norm_z
        else:
            x_i = z_thresh

        # FISTA update
        s_new = (1.0 + np.sqrt(1.0 + 4.0 * s**2)) / 2.0
        y_i = x_i + ((s - 1.0) / s_new) * (x_i - x_old)
        s = s_new

        # Convergence check
        if np.linalg.norm(x_i - x_old) < tol:
            break
            
    return x_i

class Solver:
    def solve(self, problem: Dict, **kwargs) -> Any:
        """
        Solves the sparse PCA problem.
        1. Uses eigsh for efficient partial eigendecomposition.
        2. Solves the relaxed problem using FISTA for faster convergence.
        3. JIT-compiles the FISTA solver using Numba for speed.
        """
        A = np.array(problem["covariance"])
        n_components = int(problem["n_components"])
        sparsity_param = float(problem["sparsity_param"])
        n = A.shape[0]

        if n_components == 0:
            return {"components": [], "explained_variance": []}

        # Eigendecomposition
        try:
            # Use eigsh for k < n-1 as it's faster for partial decomposition.
            # For k >= n-1, eigh is often more stable and sometimes faster.
            if n_components >= n - 1:
                eigvals, eigvecs = np.linalg.eigh(A)
                idx = np.argsort(eigvals)[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                eigvals = eigvals[:n_components]
                eigvecs = eigvecs[:, :n_components]
            else:
                # eigsh finds the k largest eigenvalues. 'LM' = Largest Magnitude.
                eigvals, eigvecs = eigsh(A, k=n_components, which='LM', tol=1e-6)
                # Re-sort as eigsh doesn't guarantee order for close eigenvalues
                idx = np.argsort(eigvals)[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
        except (np.linalg.LinAlgError, RuntimeError):
            # Fallback for decomposition failure
            return {"components": [], "explained_variance": []}

        # Filter out small non-positive eigenvalues for numerical stability
        pos_indices = eigvals > 1e-9
        eigvals = eigvals[pos_indices]
        eigvecs = eigvecs[:, pos_indices]

        k = len(eigvals)

        # Construct the target matrix B
        B = np.zeros((n, n_components))
        if k > 0:
            B[:, :k] = eigvecs * np.sqrt(eigvals)

        # Initialize the solution matrix X
        X = np.zeros((n, n_components))

        # Parameters for the FISTA solver
        max_iter = 1000
        tol = 1e-6
        # Lipschitz constant of grad(f) is 2, so step size t < 1/L = 0.5
        t = 0.49

        # Solve for each component independently using the jitted FISTA solver
        for i in range(n_components):
            b_i = B[:, i]
            X[:, i] = _solve_component_fista(b_i, sparsity_param, t, max_iter, tol)

        # Calculate explained variance for each component
        explained_variance = [float(X[:, i].T @ A @ X[:, i]) for i in range(n_components)]

        return {"components": X.tolist(), "explained_variance": explained_variance}
        return {"components": X.T.tolist(), "explained_variance": explained_variance}