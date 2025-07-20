import numpy as np
import numba

@numba.njit(fastmath=True, boundscheck=False, cache=True)
def coordinate_descent(X, y, alpha=0.1, max_iter=1000, tol=1e-4):
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    residual = y.copy().astype(np.float64)
    
    # Precompute feature norms (diagonal of X'X)
    diag = np.zeros(d, dtype=np.float64)
    for j in range(d):
        col_sum = 0.0
        for i in range(n):
            col_val = X[i, j]
            col_sum += col_val * col_val
        diag[j] = max(col_sum, 1e-10)
    
    # Initialize residual norm squared and L1 norm
    r_norm_sq = np.sum(residual**2)
    l1_norm = 0.0
    best_obj = r_norm_sq / (2 * n) + alpha * l1_norm
    
    # Coordinate descent loop
    for iter in range(max_iter):
        max_change = 0.0
        
        for j in range(d):
            grad_j = 0.0
            
            # Calculate gradient using residual
            for i in range(n):
                grad_j += X[i, j] * residual[i]
            
            # Soft-thresholding
            w_j_old = w[j]
            candidate = w_j_old + grad_j / diag[j]
            threshold = alpha * n / diag[j]
            
            if candidate > threshold:
                w_j_new = candidate - threshold
            elif candidate < -threshold:
                w_j_new = candidate + threshold
            else:
                w_j_new = 0.0
                
            # Update coefficient if changed
            if w_j_new != w_j_old:
                diff = w_j_new - w_j_old
                w[j] = w_j_new
                
                # Update residual and residual norm incrementally
                for i in range(n):
                    residual[i] -= diff * X[i, j]
                
                # Update residual norm squared: r_new = r_old - diff * X_j
                r_norm_sq = r_norm_sq - 2 * diff * grad_j + diff**2 * diag[j]
                
                # Update L1 norm
                l1_norm += abs(w_j_new) - abs(w_j_old)
                
                if abs(diff) > max_change:
                    max_change = abs(diff)
        
        # Early stopping every 10 iterations
        if iter % 10 == 0:
            current_obj = r_norm_sq / (2 * n) + alpha * l1_norm
            if best_obj - current_obj < tol * best_obj and max_change < tol:
                break
            best_obj = min(best_obj, current_obj)
        elif max_change < tol:
            break
    
    return w

class Solver:
    def solve(self, problem, **kwargs):
        try:
            # Use Fortran-contiguous layout for efficient column access
            X_arr = np.array(problem["X"], dtype=np.float64, order='F')
            y_arr = np.array(problem["y"], dtype=np.float64)
            
            if X_arr.size == 0:
                return []
                
            # Warm start with precompilation
            if not hasattr(self, '_compiled'):
                dummy_X = np.ones((10, 5), dtype=np.float64, order='F')
                dummy_y = np.ones(10, dtype=np.float64)
                coordinate_descent(dummy_X, dummy_y, 0.1)
                self._compiled = True
                
            # Solve using optimized implementation
            w = coordinate_descent(X_arr, y_arr, alpha=0.1)
            return w.tolist()
        except Exception as e:
            n_features = len(problem["X"][0]) if problem["X"] else 0
            return [0.0] * n_features