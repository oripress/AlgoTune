from typing import Any, Dict
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.exceptions import NotFittedError

class Solver:
    """
    Solver for Kernel Density Estimation (KDE) tasks.

    Uses sklearn.neighbors.KernelDensity to ensure numerical results match
    the reference implementation.
    """

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        try:
            # Required keys
            for key in ("data_points", "query_points", "kernel", "bandwidth"):
                if key not in problem:
                    raise KeyError(key)

            X = np.asarray(problem["data_points"], dtype=float)
            X_q = np.asarray(problem["query_points"], dtype=float)
            kernel = problem["kernel"]
            bandwidth = problem["bandwidth"]

            # Basic validation
            if X.ndim != 2 or X_q.ndim != 2:
                raise ValueError("Data points or query points are not 2D arrays.")
            if X.shape[0] == 0:
                raise ValueError("No data points provided.")
            if X_q.shape[0] == 0:
                # Return empty list if no query points
                return {"log_density": []}
            if X.shape[1] != X_q.shape[1]:
                raise ValueError("Data points and query points have different dimensions.")
            if not isinstance(bandwidth, (float, int)) or bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")
            bandwidth = float(bandwidth)

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