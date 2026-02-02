import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from numba import njit

@njit(cache=True, fastmath=True)
def build_B_and_process(eigvals, eigvecs, n, n_components, threshold):
    k = len(eigvals)
    components = np.zeros((n, n_components), dtype=np.float64)
    m = min(k, n_components)
    
    for j in range(m):
        scale = np.sqrt(eigvals[j])
        norm_sq = 0.0
        for i in range(n):
            val = eigvecs[i, j] * scale
            if val > threshold:
                new_val = val - threshold
            elif val < -threshold:
                new_val = val + threshold
            else:
                new_val = 0.0
            components[i, j] = new_val
            norm_sq += new_val * new_val
        
        if norm_sq > 1.0:
            inv_norm = 1.0 / np.sqrt(norm_sq)
            for i in range(n):
                components[i, j] *= inv_norm
    
    return components

class Solver:
    def solve(self, problem, **kwargs):
        cov = problem["covariance"]
        if isinstance(cov, np.ndarray):
            A = np.ascontiguousarray(cov, dtype=np.float64)
        else:
            A = np.array(cov, dtype=np.float64, order='C')
        
        n_components = int(problem["n_components"])
        sparsity_param = float(problem["sparsity_param"])
        
        n = A.shape[0]
        threshold = sparsity_param / 2.0
        
        # Use eigsh for truncated eigendecomposition when beneficial
        use_eigsh = n_components < n // 2 and n > 15
        
        if use_eigsh:
            try:
                eigvals, eigvecs = eigsh(A, k=n_components, which='LM', tol=1e-6, 
                                         ncv=min(n, max(2*n_components+1, 20)))
                idx = np.argsort(eigvals)[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
            except:
                use_eigsh = False
        
        if not use_eigsh:
            if n_components >= n:
                eigvals, eigvecs = eigh(A, overwrite_a=True, check_finite=False)
            else:
                eigvals, eigvecs = eigh(A, subset_by_index=(n - n_components, n - 1), 
                                        overwrite_a=True, check_finite=False)
            eigvals = eigvals[::-1]
            eigvecs = eigvecs[:, ::-1]
        
        # Filter positive eigenvalues
        pos_mask = eigvals > 0
        eigvals = np.ascontiguousarray(eigvals[pos_mask])
        eigvecs = np.ascontiguousarray(eigvecs[:, pos_mask])
        
        # Combined B construction and soft-threshold + projection
        components = build_B_and_process(eigvals, eigvecs, n, n_components, threshold)
        
        # Explained variance
        Ax = A @ components
        explained_variance = np.einsum('ij,ij->j', components, Ax)
        
        return {"components": components.tolist(), "explained_variance": explained_variance.tolist()}