import numpy as np
from sklearn.neighbors import KernelDensity
import numba
from math import pi, exp, log, cos, sqrt

@numba.jit(nopython=True, parallel=True, fastmath=True)
def kde_gaussian(X, X_q, h, dim, log_const):
    n_points = X.shape[0]
    n_query = X_q.shape[0]
    dims = X.shape[1]
    inv_2h2 = 1.0 / (2.0 * h * h)
    log_norm = log_const - log(n_points) - dims * log(h)
    
    results = np.empty(n_query, dtype=np.float64)
    
    for i in range(n_query):
        sum_val = 0.0
        
        for j in range(n_points):
            d_sq = 0.0
            for k in range(dims):
                diff = X[j, k] - X_q[i, k]
                d_sq += diff * diff
            
            sum_val += exp(-d_sq * inv_2h2)
            
        if sum_val <= 0.0:
            results[i] = -np.inf
        else:
            results[i] = log(sum_val) + log_norm
        
    return results

@numba.jit(nopython=True, parallel=True, fastmath=True)
def kde_exponential(X, X_q, h, dim, log_const):
    n_points = X.shape[0]
    n_query = X_q.shape[0]
    dims = X.shape[1]
    inv_h = 1.0 / h
    log_norm = log_const - log(n_points) - dims * log(h)
    
    results = np.empty(n_query, dtype=np.float64)
    
    for i in range(n_query):
        sum_val = 0.0
        
        for j in range(n_points):
            d_sq = 0.0
            for k in range(dims):
                diff = X[j, k] - X_q[i, k]
                d_sq += diff * diff
            d = sqrt(d_sq)
            
            sum_val += exp(-d * inv_h)
            
        if sum_val <= 0.0:
            results[i] = -np.inf
        else:
            results[i] = log(sum_val) + log_norm
        
    return results
@numba.jit(nopython=True, parallel=True, fastmath=True)
def kde_compact(X, X_q, h, dim, log_const, kernel_type):
    # kernel_type: 1=tophat, 2=epanechnikov, 4=linear, 5=cosine
    n_points = X.shape[0]
    n_query = X_q.shape[0]
    dims = X.shape[1]
    h_sq = h * h
    inv_h = 1.0 / h
    log_norm = log_const - log(n_points) - dims * log(h)
    
    results = np.empty(n_query, dtype=np.float64)
    
    for i in range(n_query):
        total_val = 0.0
        
        for j in range(n_points):
            dist_sq = 0.0
            for k in range(dims):
                diff = X[j, k] - X_q[i, k]
                dist_sq += diff * diff
            
            if dist_sq >= h_sq:
                continue
            
            val = 0.0
            if kernel_type == 1: # Tophat
                val = 1.0
            elif kernel_type == 2: # Epanechnikov
                val = 1.0 - dist_sq / h_sq
            elif kernel_type == 4: # Linear
                dist = sqrt(dist_sq)
                val = 1.0 - dist * inv_h
            elif kernel_type == 5: # Cosine
                dist = sqrt(dist_sq)
                val = cos(pi * dist * inv_h * 0.5)
            
            total_val += val
        
        if total_val <= 0.0:
            results[i] = -np.inf
        else:
            results[i] = log(total_val) + log_norm
            
    return results

class Solver:
    def __init__(self):
        # Warmup
        X = np.array([[0.0]], dtype=np.float64)
        X_q = np.array([[0.0]], dtype=np.float64)
        kde_gaussian(X, X_q, 1.0, 1, 0.0)
        kde_exponential(X, X_q, 1.0, 1, 0.0)
        kde_compact(X, X_q, 1.0, 1, 0.0, 1)

    def solve(self, problem, **kwargs):
        try:
            if "data_points" not in problem or "query_points" not in problem:
                 return {"error": "Missing keys"}
            
            X = np.ascontiguousarray(np.array(problem["data_points"], dtype=np.float64))
            X_q = np.ascontiguousarray(np.array(problem["query_points"], dtype=np.float64))
            
            if X.ndim != 2 or X_q.ndim != 2:
                raise ValueError("Data points or query points are not 2D arrays.")
            if X.shape[0] == 0:
                raise ValueError("No data points provided.")
            if X_q.shape[0] == 0:
                return {"log_density": []}
            if X.shape[1] != X_q.shape[1]:
                raise ValueError("Data points and query points have different dimensions.")
                
            kernel = problem.get("kernel", "gaussian")
            bandwidth = problem.get("bandwidth", 1.0)
            
            if bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")
            bandwidth = float(bandwidth)
            
            dims = X.shape[1]
            
            # Get normalization constant
            dummy_kde = KernelDensity(kernel=kernel, bandwidth=1.0)
            dummy_kde.fit(np.zeros((1, dims)))
            log_k0 = dummy_kde.score_samples(np.zeros((1, dims)))[0]
            
            if kernel == 'gaussian':
                res = kde_gaussian(X, X_q, bandwidth, dims, log_k0)
            elif kernel == 'exponential':
                res = kde_exponential(X, X_q, bandwidth, dims, log_k0)
            elif kernel == 'tophat':
                res = kde_compact(X, X_q, bandwidth, dims, log_k0, 1)
            elif kernel == 'epanechnikov':
                res = kde_compact(X, X_q, bandwidth, dims, log_k0, 2)
            elif kernel == 'linear':
                res = kde_compact(X, X_q, bandwidth, dims, log_k0, 4)
            elif kernel == 'cosine':
                res = kde_compact(X, X_q, bandwidth, dims, log_k0, 5)
            else:
                raise ValueError(f"Unknown kernel: {kernel}")
                
            return {"log_density": res.tolist()}
            
        except Exception as e:
            return {"error": f"Error: {e}"}