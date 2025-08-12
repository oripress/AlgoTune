import numpy as np
from scipy.optimize import leastsq

class Solver:
    def _safe_exp(self, z):
        """Exponentiation clipped to avoid overflow."""
        return np.exp(np.clip(z, -50.0, 50.0))
    
    def solve(self, problem, **kwargs):
        x_data = np.asarray(problem["x_data"])
        y_data = np.asarray(problem["y_data"])
        model_type = problem["model_type"]
        
        if model_type == "polynomial":
            degree = problem["degree"]
            
            def residual(p):
                return y_data - np.polyval(p, x_data)
            
            guess = np.ones(degree + 1)
            
        elif model_type == "exponential":
            def residual(p):
                a, b, c = p
                return y_data - (a * self._safe_exp(b * x_data) + c)
            
            guess = np.array([1.0, 0.05, 0.0])
            
        elif model_type == "logarithmic":
            def residual(p):
                a, b, c, d = p
                return y_data - (a * np.log(b * x_data + c) + d)
            
            guess = np.array([1.0, 1.0, 1.0, 0.0])
            
        elif model_type == "sigmoid":
            def residual(p):
                a, b, c, d = p
                return y_data - (a / (1 + self._safe_exp(-b * (x_data - c))) + d)
            
            guess = np.array([3.0, 0.5, np.median(x_data), 0.0])
            
        elif model_type == "sinusoidal":
            def residual(p):
                a, b, c, d = p
                return y_data - (a * np.sin(b * x_data + c) + d)
            
            guess = np.array([2.0, 1.0, 0.0, 0.0])
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Use leastsq to match the reference implementation
        params_opt, cov_x, info, mesg, ier = leastsq(
            residual, guess, full_output=True, maxfev=2000
        )
        
        return {"params": params_opt.tolist()}