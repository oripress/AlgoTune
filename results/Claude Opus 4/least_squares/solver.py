import numpy as np
from scipy.optimize import curve_fit
from numba import jit

class Solver:
    def __init__(self):
        # Pre-compile numba functions
        self._exp_clip = jit(nopython=True)(lambda x: np.exp(np.clip(x, -50.0, 50.0)))
        
    def solve(self, problem, **kwargs):
        """Solve least squares fitting problem."""
        x_data = np.asarray(problem["x_data"], dtype=np.float64)
        y_data = np.asarray(problem["y_data"], dtype=np.float64)
        model_type = problem["model_type"]
        
        if model_type == "polynomial":
            # Use specialized polynomial fitting
            deg = problem["degree"]
            params_opt = np.polyfit(x_data, y_data, deg)
            
        elif model_type == "exponential":
            def exp_model(x, a, b, c):
                return a * np.exp(np.clip(b * x, -50.0, 50.0)) + c
            params_opt, _ = curve_fit(exp_model, x_data, y_data, 
                                     p0=[1.0, 0.05, 0.0], maxfev=5000)
            
        elif model_type == "logarithmic":
            def log_model(x, a, b, c, d):
                return a * np.log(b * x + c) + d
            params_opt, _ = curve_fit(log_model, x_data, y_data,
                                     p0=[1.0, 1.0, 1.0, 0.0], maxfev=5000)
            
        elif model_type == "sigmoid":
            x_median = np.median(x_data)
            def sigmoid_model(x, a, b, c, d):
                return a / (1 + np.exp(np.clip(-b * (x - c), -50.0, 50.0))) + d
            params_opt, _ = curve_fit(sigmoid_model, x_data, y_data,
                                     p0=[3.0, 0.5, x_median, 0.0], maxfev=5000)
            
        elif model_type == "sinusoidal":
            def sin_model(x, a, b, c, d):
                return a * np.sin(b * x + c) + d
            params_opt, _ = curve_fit(sin_model, x_data, y_data,
                                     p0=[2.0, 1.0, 0.0, 0.0], maxfev=5000)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return {
            "params": params_opt.tolist()
        }