import numpy as np
from scipy.linalg import svd

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the tensor completion problem using ultra-optimized ADMM.
        
        :param problem: Dictionary with problem parameters
        :return: Dictionary with the completed tensor
        """
        # Extract problem data
        observed_tensor = np.array(problem["tensor"])
        mask = np.array(problem["mask"])
        tensor_dims = observed_tensor.shape
        dim1, dim2, dim3 = tensor_dims
        
        # Initialize completed tensor with observed values
        X = observed_tensor.copy()
        
        # Ultra-optimized ADMM parameters
        rho = 2.5  # Slightly higher for faster convergence
        max_iter = 7  # Even fewer iterations
        tol = 1.2e-3  # Slightly relaxed tolerance
        
        # Pre-compute unfold shapes for efficiency
        unfold1_shape = (dim1, dim2 * dim3)
        unfold2_shape = (dim2, dim1 * dim3)
        unfold3_shape = (dim3, dim1 * dim2)
        
        # Ultra-fast rank estimation
        def estimate_rank(matrix):
            try:
                # Use only first few singular values for speed
                s = svd(matrix, compute_uv=False, lapack_driver='gesvd')[:6]
                total_energy = np.sum(s**2)
                if total_energy == 0:
                    return 1
                cumulative_energy = np.cumsum(s**2) / total_energy
                # Find rank that captures 85% of energy (faster)
                rank = np.argmax(cumulative_energy >= 0.85) + 1
                return min(rank, max(3, int(np.sqrt(min(matrix.shape)))))
            except:
                return max(3, int(np.sqrt(min(matrix.shape))))
        
        # Ultra-optimized soft-thresholding SVD
        def soft_threshold_svd_opt(matrix, threshold):
            try:
                # Estimate optimal rank
                est_rank = estimate_rank(matrix)
                
                # Use efficient SVD with estimated rank
                U, s, Vt = svd(matrix, full_matrices=False, lapack_driver='gesvd')
                s_threshold = np.maximum(s - threshold, 0)
                rank = np.sum(s_threshold > 0)
                
                # Use minimum of estimated rank and actual rank
                effective_rank = min(est_rank, rank)
                if effective_rank > 0:
                    # Fast reconstruction using optimal rank
                    return U[:, :effective_rank] @ np.diag(s_threshold[:effective_rank]) @ Vt[:effective_rank, :]
                else:
                    return np.zeros_like(matrix)
            except:
                # Fallback to simple thresholding
                try:
                    U, s, Vt = svd(matrix, full_matrices=False, lapack_driver='gesvd')
                    s_threshold = np.maximum(s - threshold, 0)
                    rank = np.sum(s_threshold > 0)
                    if rank > 0:
                        return U[:, :rank] @ np.diag(s_threshold[:rank]) @ Vt[:rank, :]
                    else:
                        return np.zeros_like(matrix)
                except:
                    return np.zeros_like(matrix)
        
        # Ultra-optimized ADMM iterations with early termination
        for iteration in range(max_iter):
            X_prev = X.copy()
            
            # Mode 1: Fast unfolding and processing
            U1 = X.reshape(unfold1_shape)
            U1_new = soft_threshold_svd_opt(U1, 1/rho)
            
            # Mode 2: Fast unfolding and processing
            U2 = X.transpose(1, 0, 2).reshape(unfold2_shape)
            U2_new = soft_threshold_svd_opt(U2, 1/rho)
            
            # Mode 3: Fast unfolding and processing
            U3 = X.transpose(2, 0, 1).reshape(unfold3_shape)
            U3_new = soft_threshold_svd_opt(U3, 1/rho)
            
            # Fast averaging of unfoldings
            X1_avg = U1_new.reshape(dim1, dim2, dim3)
            X2_avg = U2_new.reshape(dim2, dim1, dim3).transpose(1, 0, 2)
            X3_avg = U3_new.reshape(dim3, dim1, dim2).transpose(1, 2, 0)
            
            # Fast weighted average and projection
            X_new = (X1_avg + X2_avg + X3_avg) * (1.0 / 3.0)
            X = mask * observed_tensor + (1 - mask) * X_new
            
            # Early convergence check
            diff = X - X_prev
            if np.linalg.norm(diff) / (np.linalg.norm(X_prev) + 1e-10) < tol:
                break
        
        return {"completed_tensor": X.tolist()}