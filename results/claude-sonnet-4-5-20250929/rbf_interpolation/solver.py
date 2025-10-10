import numpy as np
from scipy.interpolate import RBFInterpolator

class Solver:
    def solve(self, problem, **kwargs):
        """
        Optimized RBF interpolation using scipy with minimal overhead.
        """
        # Use C-contiguous arrays directly without copies
        x_train = np.ascontiguousarray(problem["x_train"], dtype=np.float64)
        y_train = np.ascontiguousarray(problem["y_train"], dtype=np.float64).ravel()
        x_test = np.ascontiguousarray(problem["x_test"], dtype=np.float64)

        rbf_config = problem["rbf_config"]
        
        # Create interpolator
        rbf_interpolator = RBFInterpolator(
            x_train, 
            y_train, 
            kernel=rbf_config["kernel"], 
            epsilon=rbf_config["epsilon"], 
            smoothing=rbf_config["smoothing"]
        )

        # Evaluate on test points
        y_pred = rbf_interpolator(x_test)

        return {"y_pred": y_pred.tolist()}