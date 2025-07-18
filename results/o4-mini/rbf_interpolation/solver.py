import numpy as np
from scipy.interpolate import RBFInterpolator

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        """
        RBF interpolation using SciPy's RBFInterpolator.
        """
        # Prepare data
        x_train = np.asarray(problem["x_train"], float)
        y_train = np.asarray(problem["y_train"], float).ravel()
        x_test = np.asarray(problem["x_test"], float)

        # Configuration parameters
        rbf_config = problem.get("rbf_config", {})
        kernel = rbf_config.get("kernel", "thin_plate_spline")
        epsilon = rbf_config.get("epsilon", None)
        smoothing = rbf_config.get("smoothing", 0.0)

        # Build interpolator and predict
        rbf = RBFInterpolator(x_train, y_train, kernel=kernel,
                              epsilon=epsilon, smoothing=smoothing)
        y_pred = rbf(x_test)

        return {"y_pred": y_pred.tolist(), "rbf_config": rbf_config}