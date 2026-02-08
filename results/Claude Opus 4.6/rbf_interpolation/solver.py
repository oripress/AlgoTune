import numpy as np
from scipy.interpolate import RBFInterpolator

class Solver:
    def solve(self, problem, **kwargs):
        x_train = np.asarray(problem["x_train"], dtype=np.float64)
        y_train = np.asarray(problem["y_train"], dtype=np.float64).ravel()
        x_test = np.asarray(problem["x_test"], dtype=np.float64)
        
        config = problem.get("rbf_config", {})
        kernel = config.get("kernel", "thin_plate_spline")
        epsilon = config.get("epsilon", 1.0)
        smoothing = config.get("smoothing", 0.0)
        
        rbf = RBFInterpolator(x_train, y_train, kernel=kernel, epsilon=epsilon, smoothing=smoothing)
        y_pred = rbf(x_test)
        
        return {"y_pred": y_pred.tolist()}