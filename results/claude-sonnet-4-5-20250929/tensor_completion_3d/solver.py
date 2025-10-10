import numpy as np
from sklearn.utils.extmath import randomized_svd

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the tensor completion problem using optimized ADMM with randomized SVD.
        """
        observed_tensor = np.array(problem["tensor"], dtype=np.float64)
        mask = np.array(problem["mask"], dtype=bool)
        tensor_dims = observed_tensor.shape
        
        dim1, dim2, dim3 = tensor_dims
        
        # Unfold along mode 1
        unfolding1 = observed_tensor.reshape(dim1, dim2 * dim3)
        mask1 = mask.reshape(dim1, dim2 * dim3)
        
        # Solve matrix completion on unfolding1
        X = self._admm_matrix_completion(unfolding1, mask1)
        
        completed_tensor = X.reshape(tensor_dims)
        return {"completed_tensor": completed_tensor.tolist()}
    
    def _admm_matrix_completion(self, M, mask, max_iter=75, rho=3.5):
        """
        Matrix completion using ADMM with randomized SVD for maximum speedup.
        """
        m, n = M.shape
        rank_est = min(6, min(m, n) - 1)
        
        # Initialize
        X = M.copy()
        Z = np.zeros_like(M, dtype=np.float64)
        U = np.zeros_like(M, dtype=np.float64)
        
        tau = 1.0 / rho
        
        for iteration in range(max_iter):
            # Update X (projection onto constraint set) - optimized
            X = Z - U
            X[mask] = M[mask]
            
            # Update Z using soft-thresholding with randomized SVD
            Y = X + U
            
            # Use randomized SVD for speed
            if min(m, n) > rank_est + 2:
                # Randomized SVD with minimal iterations
                Uy, s, Vt = randomized_svd(Y, n_components=rank_est, n_iter=2, random_state=42)
                s_thresh = np.maximum(s - tau, 0)
                Z = (Uy * s_thresh) @ Vt
            else:
                # Full SVD for small matrices
                Uy, s, Vt = np.linalg.svd(Y, full_matrices=False)
                s_thresh = np.maximum(s - tau, 0)
                Z = (Uy * s_thresh) @ Vt
            
            # Update dual variable
            U += X - Z
        
        return X