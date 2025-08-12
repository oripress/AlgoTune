from typing import Any, Dict, Optional

import numpy as np
from scipy.interpolate import RBFInterpolator

def _select_neighbors(
    n_samples: int,
    n_dims: int,
    n_test: int,
    kernel: Optional[str],
    smoothing: Optional[float],
) -> Optional[int]:
    """
    Choose a conservative number of neighbors only for large, smooth, localized kernels
    to ensure predictions remain close to the global solution while gaining speed.
    """
    # Only consider localized kernels and smooth problems
    if kernel not in ("gaussian", "inverse_multiquadric"):
        return None
    if smoothing is None or smoothing <= 0:
        return None

    # Avoid high-dimensional cases where locality is less reliable
    if n_dims > 6:
        return None

    # Only use neighbors for large sample sizes, where global solve is expensive
    if n_samples < 1500:
        return None

    # Use a fraction of samples, capped to keep computation fast but accurate
    base = int(max(128, 0.2 * n_samples))
    base += 8 * n_dims  # slight increase with dimensionality

    k = min(800, n_samples, base)
    return k

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the RBF interpolation problem using scipy.interpolate.RBFInterpolator.
        Use a conservative local-neighbors strategy only when it is likely to be
        both faster and sufficiently accurate.
        """
        x_train = np.asarray(problem["x_train"], dtype=float)
        y_train = np.asarray(problem["y_train"], dtype=float).ravel()
        x_test = np.asarray(problem["x_test"], dtype=float)

        # Ensure C-contiguous arrays for faster distance computations
        x_train = np.ascontiguousarray(x_train, dtype=float)
        x_test = np.ascontiguousarray(x_test, dtype=float)
        y_train = np.ascontiguousarray(y_train, dtype=float)

        rbf_config = problem.get("rbf_config") or {}
        kernel = rbf_config.get("kernel")
        epsilon = rbf_config.get("epsilon")
        smoothing = rbf_config.get("smoothing")

        # Fast path: exact interpolation at training points
        # If smoothing is zero and x_test exactly equals x_train, return y_train directly.
        if (smoothing == 0 or smoothing == 0.0) and x_test.shape == x_train.shape and x_test.size != 0:
            if np.array_equal(x_test, x_train):
                return {"y_pred": y_train.tolist()}

        n_samples, n_dims = x_train.shape
        n_test = x_test.shape[0]
        neighbors = _select_neighbors(n_samples, n_dims, n_test, kernel, smoothing)

        rbf_interpolator = RBFInterpolator(
            x_train,
            y_train,
            kernel=kernel,
            epsilon=epsilon,
            smoothing=smoothing,
            neighbors=neighbors,
        )

        # Deduplicate test points only for large test sets to avoid overhead
        if x_test.size == 0:
            y_pred = np.empty((0,), dtype=float)
        elif n_test >= 2000:
            try:
                unique_x, inv = np.unique(x_test, axis=0, return_inverse=True)
                if unique_x.shape[0] < x_test.shape[0]:
                    y_unique = rbf_interpolator(unique_x)
                    y_pred = y_unique[inv]
                else:
                    y_pred = rbf_interpolator(x_test)
            except Exception:
                # Fallback to direct evaluation if uniqueness on axis=0 is unsupported
                y_pred = rbf_interpolator(x_test)
        else:
            y_pred = rbf_interpolator(x_test)

        return {
            "y_pred": y_pred.tolist(),
        }