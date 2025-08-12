import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.exceptions import NotFittedError

class Solver:
    available_kernels = ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
    
    def solve(self, problem, **kwargs):
        try:
            # Extract data from problem dictionary
            data_points = problem["data_points"]
            query_points = problem["query_points"]
            kernel = problem["kernel"]
            bandwidth = problem["bandwidth"]
            
            # Convert to arrays for dimension validation
            X = np.array(data_points, dtype=np.float64)
            X_q = np.array(query_points, dtype=np.float64)
            
            # Check array dimensions
            if X.ndim != 2 or X_q.ndim != 2:
                raise ValueError("Data points or query points are not 2D arrays.")
                
            # Handle empty data points
            if X.shape[0] == 0:
                if X_q.shape[0] == 0:
                    return {"log_density": []}
                raise ValueError("No data points provided.")
            
            # Handle empty query points
            if X_q.shape[0] == 0:
                return {"log_density": []}
            
            # Validate dimensions
            if X.shape[1] != X_q.shape[1]:
                raise ValueError("Data points and query points have different dimensions.")
            
            # Validate bandwidth
            if not isinstance(bandwidth, (float, int)) or bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")
            
            # Validate kernel
            if kernel not in self.available_kernels:
                raise ValueError(f"Unknown kernel: {kernel}")
            
            # Use optimized scikit-learn implementation
            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
            kde.fit(X)
            log_density = kde.score_samples(X_q)
            return {"log_density": log_density.tolist()}
                
        except KeyError as e:
            return {"error": f"Missing key: {e}"}
        except (ValueError, TypeError, NotFittedError, np.linalg.LinAlgError) as e:
            return {"error": f"Computation error: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}