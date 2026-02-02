import numpy as np
from typing import Any
from sklearn.neighbors import KernelDensity

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        try:
            X = np.asarray(problem["data_points"], dtype=np.float64)
            X_q = np.asarray(problem["query_points"], dtype=np.float64)
            kernel = problem["kernel"]
            bandwidth = float(problem["bandwidth"])
            
            if X.ndim != 2 or X_q.ndim != 2:
                return {"error": "Invalid dimensions"}
            if X.shape[0] == 0:
                return {"error": "No data points"}
            if X_q.shape[0] == 0:
                return {"log_density": []}
            if X.shape[1] != X_q.shape[1]:
                return {"error": "Dimension mismatch"}
            
            # Use sklearn's KernelDensity for all kernels
            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
            kde.fit(X)
            log_density = kde.score_samples(X_q)
            
            return {"log_density": log_density.tolist()}
            
        except Exception as e:
            return {"error": str(e)}