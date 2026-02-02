import numpy as np
from scipy.linalg import svd

class Solver:
    def solve(self, problem: dict) -> dict:
        # Extract problem data
        observed_tensor = np.array(problem["tensor"])
        mask = np.array(problem["mask"])
        tensor_dims = observed_tensor.shape
        
        d1, d2, d3 = tensor_dims
        
        # Initial X
        X = observed_tensor.copy()
        X[~mask] = 0.0
        
        # ADMM parameters
        rho = 1.0
        max_iter = 100
        tol = 1e-4
        
        # Helper for unfolding
        def unfold(tensor, mode):
            if mode == 0:
                return tensor.reshape(d1, d2*d3)
            elif mode == 1:
                return tensor.transpose(1, 0, 2).reshape(d2, d1*d3)
            elif mode == 2:
                return tensor.transpose(2, 0, 1).reshape(d3, d1*d2)
            return tensor
        
        # Helper for folding
        def fold(matrix, mode):
            if mode == 0:
                return matrix.reshape(d1, d2, d3)
            elif mode == 1:
                return matrix.reshape(d2, d1, d3).transpose(1, 0, 2)
            elif mode == 2:
                return matrix.reshape(d3, d1, d2).transpose(1, 2, 0)
            return matrix

        # Shapes for unfoldings
        shapes = [
            (d1, d2*d3),
            (d2, d1*d3),
            (d3, d1*d2)
        ]
        
        # Initialize Ms and Ys
        Ms = [np.zeros(s) for s in shapes]
        Ys = [np.zeros(s) for s in shapes]
        
        # Initialize Ms with unfolded X
        for i in range(3):
            Ms[i] = unfold(X, i)
            
        mask_indices = np.where(mask)
        observed_values = observed_tensor[mask_indices]
        
        for k in range(max_iter):
            X_old_norm = np.linalg.norm(X)
            
            # Update M_i
            for i in range(3):
                # Target = Unfold_i(X) + Y_i / rho
                X_i = unfold(X, i)
                target = X_i + Ys[i] * (1.0/rho)
                
                # SVT
                thresh = 1.0 / rho
                
                try:
                    # overwrite_a=True allows destroying target for speed
                    u, s, vt = svd(target, full_matrices=False, overwrite_a=True, check_finite=False)
                    
                    # Soft thresholding
                    s = np.maximum(s - thresh, 0)
                    
                    # Reconstruct M_i
                    Ms[i] = (u * s) @ vt
                except Exception:
                    pass
            
            # Update X
            sum_tensor = np.zeros_like(X)
            for i in range(3):
                term = Ms[i] - Ys[i] * (1.0/rho)
                sum_tensor += fold(term, i)
            
            X = sum_tensor / 3.0
            
            # Enforce constraints
            X[mask_indices] = observed_values
            
            # Update Y_i
            for i in range(3):
                X_i = unfold(X, i)
                Ys[i] += rho * (X_i - Ms[i])
            
            # Check convergence
            current_norm = np.linalg.norm(X)
            if k > 0 and abs(current_norm - X_old_norm) < tol * X_old_norm:
                 break
                 
        return {"completed_tensor": X.tolist()}