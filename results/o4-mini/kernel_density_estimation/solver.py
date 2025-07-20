import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.exceptions import NotFittedError

class Solver:
    def solve(self, problem, **kwargs):
        try:
            # Parse inputs
            X = np.array(problem["data_points"], dtype=float)
            X_q = np.array(problem["query_points"], dtype=float)
            kernel = problem["kernel"]
            bandwidth = problem["bandwidth"]

            # Validate shapes
            if X.ndim != 2 or X_q.ndim != 2:
                raise ValueError("Data points or query points are not 2D arrays.")
            n, d = X.shape
            m, dq = X_q.shape
            if d != dq:
                raise ValueError("Data points and query points have different dimensions.")
            if m == 0:
                return {"log_density": []}
            if n == 0:
                raise ValueError("No data points provided.")
            if not isinstance(bandwidth, (int, float)) or bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")
            if kernel not in ("gaussian", "tophat", "epanechnikov"):
                raise ValueError(f"Unknown kernel: {kernel}")

            # Fit and evaluate KDE
            kde = KernelDensity(kernel=kernel, bandwidth=float(bandwidth))
            kde.fit(X)
            log_density = kde.score_samples(X_q)
            return {"log_density": log_density.tolist()}

        except KeyError as e:
            return {"error": f"Missing key: {e}"}
        except (ValueError, TypeError, NotFittedError, np.linalg.LinAlgError) as e:
            return {"error": f"Computation error: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}