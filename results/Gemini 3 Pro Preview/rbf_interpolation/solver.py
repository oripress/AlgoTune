import numpy as np
from scipy.interpolate import RBFInterpolator
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        x_train = np.asarray(problem["x_train"], float)
        y_train = np.asarray(problem["y_train"], float).ravel()
        x_test = np.asarray(problem["x_test"], float)

        rbf_config = problem.get("rbf_config", {})
        kernel = rbf_config.get("kernel", "thin_plate_spline")
        epsilon = rbf_config.get("epsilon", 1.0)
        smoothing = rbf_config.get("smoothing", 0.0)

        # RBFInterpolator defaults:
        # degree is None by default.
        # If kernel in ['thin_plate_spline', 'cubic', 'quintic'], degree defaults to 1.
        # Else degree defaults to -1 (no polynomial).
        
        # However, the reference implementation just calls RBFInterpolator directly.
        # We should do the same or replicate its logic.
        
        rbf_interpolator = RBFInterpolator(
            x_train, y_train, kernel=kernel, epsilon=epsilon, smoothing=smoothing
        )

        y_pred = rbf_interpolator(x_test)

        solution = {
            "y_pred": y_pred.tolist(),
            "rbf_config": rbf_config
        }

        return solution