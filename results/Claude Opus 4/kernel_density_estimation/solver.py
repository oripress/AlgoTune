import numpy as np
from typing import Any
from numba import jit
import math

@jit(nopython=True, fastmath=True)
def gaussian_kde_numba(X, X_q, bandwidth, n, m, d):
    """Fast gaussian KDE using numba."""
    log_norm = -0.5 * d * np.log(2 * np.pi) - d * np.log(bandwidth) - np.log(n)
    inv_h2 = 1.0 / (2 * bandwidth * bandwidth)
    
    log_density = np.empty(m)
    
    for i in range(m):
        # Compute log sum exp for numerical stability
        max_log = -np.inf
        
        # First pass: find maximum
        for j in range(n):
            sq_dist = 0.0
            for k in range(d):
                diff = X[j, k] - X_q[i, k]
                sq_dist += diff * diff
            log_kernel = -sq_dist * inv_h2
            if log_kernel > max_log:
                max_log = log_kernel
        
        # Second pass: compute sum
        if max_log == -np.inf:
            log_density[i] = -np.inf
        else:
            sum_exp = 0.0
            for j in range(n):
                sq_dist = 0.0
                for k in range(d):
                    diff = X[j, k] - X_q[i, k]
                    sq_dist += diff * diff
                log_kernel = -sq_dist * inv_h2
                sum_exp += np.exp(log_kernel - max_log)
            
            log_density[i] = log_norm + max_log + np.log(sum_exp)
    
    return log_density

@jit(nopython=True, fastmath=True)
def tophat_kde_numba(X, X_q, bandwidth, n, m, d, volume):
    """Fast tophat KDE using numba."""
    log_norm = -np.log(n * volume)
    log_density = np.empty(m)
    bandwidth_sq = bandwidth * bandwidth
    
    for i in range(m):
        count = 0
        for j in range(n):
            sq_dist = 0.0
            for k in range(d):
                diff = X[j, k] - X_q[i, k]
                sq_dist += diff * diff
            
            if sq_dist <= bandwidth_sq:
                count += 1
        
        if count > 0:
            log_density[i] = log_norm + np.log(count)
        else:
            log_density[i] = -np.inf
    
    return log_density

@jit(nopython=True, fastmath=True)
def epanechnikov_kde_numba(X, X_q, bandwidth, n, m, d, c_d):
    """Fast Epanechnikov KDE using numba."""
    log_density = np.empty(m)
    inv_bandwidth_sq = 1.0 / (bandwidth * bandwidth)
    norm_factor = 1.0 / (n * bandwidth**d)
    
    for i in range(m):
        kernel_sum = 0.0
        
        for j in range(n):
            sq_dist = 0.0
            for k in range(d):
                diff = X[j, k] - X_q[i, k]
                sq_dist += diff * diff
            
            u = sq_dist * inv_bandwidth_sq
            if u < 1.0:
                kernel_sum += c_d * (1.0 - u)
        
        if kernel_sum > 0:
            log_density[i] = np.log(kernel_sum * norm_factor)
        else:
            log_density[i] = -np.inf
    
    return log_density

class Solver:
    def __init__(self):
        self.available_kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        # Precompile the numba functions
        dummy = np.array([[0.0]], dtype=np.float64)
        try:
            gaussian_kde_numba(dummy, dummy, 1.0, 1, 1, 1)
            tophat_kde_numba(dummy, dummy, 1.0, 1, 1, 1, 1.0)
            epanechnikov_kde_numba(dummy, dummy, 1.0, 1, 1, 1, 1.0)
        except:
            pass
    
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        try:
            # Extract data
            X = np.array(problem["data_points"], dtype=np.float64)
            X_q = np.array(problem["query_points"], dtype=np.float64)
            kernel = problem["kernel"]
            bandwidth = float(problem["bandwidth"])
            
            # Validate inputs
            if X.ndim != 2 or X_q.ndim != 2:
                raise ValueError("Data points or query points are not 2D arrays.")
            if X.shape[0] == 0:
                raise ValueError("No data points provided.")
            if X_q.shape[0] == 0:
                return {"log_density": []}
            if X.shape[1] != X_q.shape[1]:
                raise ValueError("Data points and query points have different dimensions.")
            if not isinstance(bandwidth, (float, int)) or bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")
            if kernel not in self.available_kernels:
                raise ValueError(f"Unknown kernel: {kernel}")
            
            n, d = X.shape
            m = X_q.shape[0]
            
            # Make sure arrays are contiguous for numba
            X = np.ascontiguousarray(X)
            X_q = np.ascontiguousarray(X_q)
            
            # Compute log density
            if kernel == 'gaussian':
                log_density = gaussian_kde_numba(X, X_q, bandwidth, n, m, d)
            elif kernel == 'tophat':
                # Compute volume of d-dimensional hypersphere
                if d == 1:
                    volume = 2 * bandwidth
                elif d == 2:
                    volume = np.pi * bandwidth * bandwidth
                else:
                    volume = (2 * np.pi**(d/2) / math.gamma(d/2 + 1)) * bandwidth**d
                log_density = tophat_kde_numba(X, X_q, bandwidth, n, m, d, volume)
            elif kernel == 'epanechnikov':
                # Volume constant for Epanechnikov kernel
                if d == 1:
                    c_d = 0.75
                elif d == 2:
                    c_d = 2 / np.pi
                else:
                    c_d = (d + 2) * math.gamma(d/2 + 1) / (2 * np.pi**(d/2))
                log_density = epanechnikov_kde_numba(X, X_q, bandwidth, n, m, d, c_d)
            else:
                # Fallback to sklearn for other kernels
                from sklearn.neighbors import KernelDensity
                kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
                kde.fit(X)
                log_density = kde.score_samples(X_q)
            
            return {"log_density": log_density.tolist()}
            
        except KeyError as e:
            return {"error": f"Missing key: {e}"}
        except (ValueError, TypeError) as e:
            return {"error": f"Computation error: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}