import numpy as np
from scipy.interpolate import RBFInterpolator
from typing import Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the RBF interpolation problem using scipy.interpolate.RBFInterpolator
        with optimized parameters for faster computation.
        
        :param problem: A dictionary representing the RBF interpolation problem.
        :return: A dictionary with the solution containing predicted values.
        """
        # Convert inputs to numpy arrays with optimal dtype and memory layout
        x_train = np.asarray(problem["x_train"], dtype=np.float64, order='C')
        y_train = np.asarray(problem["y_train"], dtype=np.float64, order='C').ravel()
        x_test = np.asarray(problem["x_test"], dtype=np.float64, order='C')
        
        rbf_config = problem.get("rbf_config", {})
        kernel = rbf_config.get("kernel", "thin_plate_spline")
        epsilon = rbf_config.get("epsilon", 1.0)
        smoothing = rbf_config.get("smoothing", 0.0)
        
        # For thin_plate_spline, use degree=1 (linear polynomial tail) for better conditioning
        degree = 1 if kernel == "thin_plate_spline" else 0
        
        # Create RBF interpolator with optimized parameters
        rbf_interpolator = RBFInterpolator(
            x_train, y_train, 
            kernel=kernel, 
            epsilon=epsilon, 
            smoothing=smoothing,
            degree=degree,
            neighbors=None  # Use exact computation for accuracy
        )
        
        # Predict using the interpolator
        y_pred = rbf_interpolator(x_test)
        
        solution = {
            "y_pred": y_pred.tolist(),
        }
        
        return solution