from typing import Any, Dict

import numpy as np
from scipy.interpolate import RBFInterpolator

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the RBF interpolation problem using scipy.interpolate.RBFInterpolator.

        This method creates an RBF interpolation function from the training data and
        evaluates it on the test points. It uses the RBF configuration parameters
        provided in the problem dictionary.

        :param problem: A dictionary representing the RBF interpolation problem,
                        which should include an "rbf_config" key with configuration parameters.
        :return: A dictionary with the solution containing:
                 - "y_pred": Predicted function values at test points.
                 - "rbf_config": Configuration parameters used for the RBF interpolator.
        """
        # Convert inputs to contiguous float arrays to avoid unnecessary copies in SciPy
        x_train = np.ascontiguousarray(np.asarray(problem["x_train"], dtype=float))
        y_train = np.ascontiguousarray(np.asarray(problem["y_train"], dtype=float).ravel())
        x_test = np.ascontiguousarray(np.asarray(problem["x_test"], dtype=float))

        rbf_config = problem.get("rbf_config") or {}
        kernel = rbf_config.get("kernel", "thin_plate_spline")
        # Allow epsilon to be None so SciPy can pick a heuristic if not provided
        epsilon = rbf_config.get("epsilon", None)
        smoothing = rbf_config.get("smoothing", 0.0)

        # Construct interpolator
        rbf_interpolator = RBFInterpolator(
            x_train, y_train, kernel=kernel, epsilon=epsilon, smoothing=smoothing
        )

        # Predict on test points
        y_pred = rbf_interpolator(x_test)

        # Ensure output is a flat list of floats
        solution = {
            "y_pred": y_pred.ravel().tolist(),
            "rbf_config": {
                "kernel": kernel,
                "epsilon": epsilon,
                "smoothing": smoothing,
            },
        }

        return solution