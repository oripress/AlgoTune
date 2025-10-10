import numpy as np
from sklearn.neighbors import KernelDensity
from typing import Any

class Solver:
    def __init__(self):
        self.available_kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        try:
            X = np.asarray(problem["data_points"], dtype=np.float64)
            X_q = np.asarray(problem["query_points"], dtype=np.float64)
            kernel = problem["kernel"]
            bandwidth = float(problem["bandwidth"])
            
            # Basic validation
            if X.ndim != 2 or X_q.ndim != 2:
                raise ValueError("Data points or query points are not 2D arrays.")
            if X.shape[0] == 0:
                raise ValueError("No data points provided.")
            if X_q.shape[0] == 0:
                return {"log_density": []}
            if X.shape[1] != X_q.shape[1]:
                raise ValueError("Data points and query points have different dimensions.")
            
            if bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")
            if kernel not in self.available_kernels:
                raise ValueError(f"Unknown kernel: {kernel}")
            # Use sklearn's KernelDensity with default algorithm for correctness
            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
            kde.fit(X)
            
            # Evaluate the log-density at query points
            log_density = kde.score_samples(X_q)
            
            return {"log_density": log_density.tolist()}
            
        except KeyError as e:
            return {"error": f"Missing key: {e}"}
        except Exception as e:
            return {"error": f"Computation error: {e}"}