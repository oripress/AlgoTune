import numpy as np
from typing import Any, Dict
from scipy.interpolate import RBFInterpolator

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the RBF interpolation problem using scipy's RBFInterpolator.
        """
        # Convert inputs to NumPy arrays
        x_train = np.asarray(problem["x_train"], dtype=float)
        y_train = np.asarray(problem["y_train"], dtype=float).ravel()
        x_test = np.asarray(problem["x_test"], dtype=float)

        # Extract RBF configuration
        cfg = problem.get("rbf_config", {})
        kernel = cfg.get("kernel", "thin_plate_spline")
        epsilon = cfg.get("epsilon", 1.0)
        smoothing = cfg.get("smoothing", 0.0)

        # Build the interpolator
        rbf = RBFInterpolator(
            x_train,
            y_train,
            kernel=kernel,
            epsilon=epsilon,
            smoothing=smoothing,
        )

        # Predict at test points
        y_pred = rbf(x_test)

        return {
            "y_pred": y_pred.tolist(),
            "rbf_config": {
                "kernel": kernel,
                "epsilon": epsilon,
                "smoothing": smoothing,
            },
        }