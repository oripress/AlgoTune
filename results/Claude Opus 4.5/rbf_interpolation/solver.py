import numpy as np
from scipy.interpolate import RBFInterpolator

class Solver:
    def solve(self, problem, **kwargs):
        x_train = problem["x_train"]
        y_train = problem["y_train"]
        x_test = problem["x_test"]
        
        # Ensure correct format without extra copies
        if not isinstance(x_train, np.ndarray):
            x_train = np.asarray(x_train, dtype=np.float64)
        if not isinstance(y_train, np.ndarray):
            y_train = np.asarray(y_train, dtype=np.float64)
        if not isinstance(x_test, np.ndarray):
            x_test = np.asarray(x_test, dtype=np.float64)
        
        y_train = y_train.ravel()

        rbf_config = problem["rbf_config"]
        kernel = rbf_config["kernel"]
        epsilon = rbf_config["epsilon"]
        smoothing = rbf_config["smoothing"]

        rbf_interpolator = RBFInterpolator(
            x_train, y_train, kernel=kernel, epsilon=epsilon, smoothing=smoothing
        )
        y_pred = rbf_interpolator(x_test)

        return {"y_pred": y_pred.tolist()}