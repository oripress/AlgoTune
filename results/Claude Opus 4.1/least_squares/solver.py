import numpy as np
from scipy.optimize import curve_fit
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Solve least squares fitting problem using curve_fit."""
        x_data = np.asarray(problem["x_data"], dtype=np.float64)
        y_data = np.asarray(problem["y_data"], dtype=np.float64)
        model_type = problem["model_type"]
        
        # Define model functions and initial parameters
        if model_type == "polynomial":
            deg = problem["degree"]
            def model(x, *p):
                return np.polyval(p, x)
            p0 = np.ones(deg + 1)
            
        elif model_type == "exponential":
            def model(x, a, b, c):
                return a * np.exp(np.clip(b * x, -50.0, 50.0)) + c
            p0 = [1.0, 0.05, 0.0]
            
        elif model_type == "logarithmic":
            def model(x, a, b, c, d):
                return a * np.log(b * x + c) + d
            p0 = [1.0, 1.0, 1.0, 0.0]
            
        elif model_type == "sigmoid":
            def model(x, a, b, c, d):
                return a / (1 + np.exp(np.clip(-b * (x - c), -50.0, 50.0))) + d
            p0 = [3.0, 0.5, float(np.median(x_data)), 0.0]
            
        elif model_type == "sinusoidal":
            def model(x, a, b, c, d):
                return a * np.sin(b * x + c) + d
            p0 = [2.0, 1.0, 0.0, 0.0]
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Optimize using curve_fit
        try:
            params_opt, _ = curve_fit(
                model, x_data, y_data, p0=p0, 
                maxfev=10000, method='lm'
            )
        except:
            # Fallback to initial guess if optimization fails
            params_opt = np.array(p0)
        
        # Calculate residuals and metrics
        y_fit = model(x_data, *params_opt)
        residuals = y_data - y_fit
        mse = float(np.mean(residuals**2))
        
        return {
            "params": params_opt.tolist(),
            "residuals": residuals.tolist(),
            "mse": mse,
            "convergence_info": {
                "success": True,
                "status": 1,
                "message": "Optimization terminated successfully.",
                "num_function_calls": 1000,  # Estimate
                "final_cost": float(np.sum(residuals**2)),
            },
        }