import numpy as np
import numba

@numba.njit(cache=True)
def _solve_all(eigvals, eigvecs, A, sparsity_param, n, n_components, n_eig):
    threshold = sparsity_param * 0.5
    components = np.empty((n, n_components))
    explained_variance = np.empty(n_components)
    
    for j in range(n_components):
        # Build B column and soft threshold in one pass
        norm_sq = 0.0
        if j < n_eig and eigvals[j] > 0.0:
            scale = np.sqrt(eigvals[j])
            for i in range(n):
                val = eigvecs[i, j] * scale
                abs_val = abs(val)
                if abs_val > threshold:
                    s = abs_val - threshold
                    if val < 0:
                        s = -s
                    components[i, j] = s
                    norm_sq += s * s
                else:
                    components[i, j] = 0.0
        else:
            for i in range(n):
                components[i, j] = 0.0
        
        # Project to unit ball if needed
        if norm_sq > 1.0:
            inv_norm = 1.0 / np.sqrt(norm_sq)
            for i in range(n):
                components[i, j] *= inv_norm
        
        # Calculate explained variance: x^T A x
        var = 0.0
        for i in range(n):
            ci = components[i, j]
            if ci == 0.0:
                continue
            ax_i = 0.0
            for l in range(n):
                ax_i += A[i, l] * components[l, j]
            var += ci * ax_i
        explained_variance[j] = var
    
    return components, explained_variance

class Solver:
    def __init__(self):
        # Warm up numba
        dummy_ev = np.zeros(1)
        dummy_evec = np.zeros((2, 1))
        dummy_A = np.eye(2)
        _solve_all(dummy_ev, dummy_evec, dummy_A, 0.1, 2, 1, 1)
    
    def solve(self, problem, **kwargs):
        cov = problem["covariance"]
        n_components = int(problem["n_components"])
        sparsity_param = float(problem["sparsity_param"])
        
        A = np.asarray(cov, dtype=np.float64)
        n = A.shape[0]
        
        # Eigendecomposition - use numpy for small, scipy partial for large
        if n > 64 and n_components < n // 2:
            from scipy.linalg import eigh as sp_eigh
            k = n_components
            eigvals, eigvecs = sp_eigh(A, subset_by_index=[n - k, n - 1])
            # Returns ascending, reverse to descending
            eigvals = eigvals[::-1].copy()
            eigvecs = eigvecs[:, ::-1].copy()
        else:
            eigvals_all, eigvecs_all = np.linalg.eigh(A)
            k = min(n, n_components)
            # Take top k (last k elements reversed)
            eigvals = eigvals_all[-k:][::-1].copy()
            eigvecs = eigvecs_all[:, -k:][:, ::-1].copy()
        
        components, explained_variance = _solve_all(
            eigvals, eigvecs, A, sparsity_param, n, n_components, len(eigvals)
        )
        
        return {
            "components": components.tolist(),
            "explained_variance": explained_variance.tolist()
        }