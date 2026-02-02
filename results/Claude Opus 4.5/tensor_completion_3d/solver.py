import numpy as np
from scipy.linalg import svd
from sklearn.utils.extmath import randomized_svd
import numba

class Solver:
    def solve(self, problem, **kwargs):
        observed_tensor = np.array(problem["tensor"], dtype=np.float64)
        mask = np.array(problem["mask"], dtype=bool)
        tensor_dims = observed_tensor.shape
        
        if mask.all():
            return {"completed_tensor": observed_tensor.tolist()}
        
        if not mask.any():
            return {"completed_tensor": np.zeros(tensor_dims).tolist()}
        
        dim1, dim2, dim3 = tensor_dims
        
        unfolding1 = observed_tensor.reshape(dim1, dim2 * dim3)
        mask1 = mask.reshape(dim1, dim2 * dim3)
        obs_vals = observed_tensor[mask]
        
        try:
            completed = soft_impute(unfolding1, mask1, obs_vals)
            return {"completed_tensor": completed.reshape(tensor_dims).tolist()}
        except Exception:
            return {"completed_tensor": []}

def soft_impute(observed, mask, obs_vals, max_iter=200, tol=1e-5):
    """Soft-Impute algorithm - faster than ADMM for matrix completion."""
    m, n = observed.shape
    min_dim = min(m, n)
    
    # Initial lambda estimate
    lam = 1.0
    
    # Initialize with observed values
    Z = observed.copy()
    
    use_randomized = min_dim > 20
    rank_est = max(2, min(min_dim // 2, 10))
    
    for iteration in range(max_iter):
        Z_old = Z.copy()
        
        # SVD and soft-thresholding
        if use_randomized and rank_est < min_dim - 2:
            try:
                U, s, Vt = randomized_svd(Z, n_components=rank_est, n_iter=2, random_state=None)
                # Check if we need more components
                if s[-1] > lam:
                    rank_est = min(rank_est * 2, min_dim - 1)
                    U, s, Vt = svd(Z, full_matrices=False, check_finite=False, lapack_driver='gesdd')
            except:
                U, s, Vt = svd(Z, full_matrices=False, check_finite=False, lapack_driver='gesdd')
        else:
            U, s, Vt = svd(Z, full_matrices=False, check_finite=False, lapack_driver='gesdd')
        
        # Soft threshold
        s_thresh = np.maximum(s - lam, 0.0)
        
        # Count non-zero and reconstruct
        nnz = np.count_nonzero(s_thresh)
        if nnz > 0:
            Z = (U[:, :nnz] * s_thresh[:nnz]) @ Vt[:nnz, :]
            rank_est = max(nnz + 2, rank_est)
        else:
            Z = np.zeros_like(observed)
        
        # Replace observed entries
        Z[mask] = obs_vals
        
        # Check convergence
        diff = np.linalg.norm(Z - Z_old, 'fro')
        norm_old = np.linalg.norm(Z_old, 'fro')
        if norm_old > 0 and diff / norm_old < tol:
            break
        
        # Adaptive lambda - decrease for faster convergence
        if iteration > 10:
            lam *= 0.95
    
    return Z