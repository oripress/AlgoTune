import numpy as np
from scipy.optimize import leastsq
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Finds the parameters of a model that best fit the data.
        """
        model_type = problem["model_type"]
        x_data = np.asarray(problem["x_data"])
        y_data = np.asarray(problem["y_data"])

        if model_type == "polynomial":
            degree = problem["degree"]
            params = np.polyfit(x_data, y_data, degree)
            return {"params": params.tolist()}

        def _safe_exp(z: np.ndarray | float) -> np.ndarray | float:
            return np.exp(np.clip(z, -50.0, 50.0))

        if model_type == "exponential":
            def r(p):
                a, b, c = p
                return y_data - (a * _safe_exp(b * x_data) + c)
            guess = np.array([1.0, 0.05, 0.0])
        elif model_type == "logarithmic":
            def r(p):
                a, b, c, d = p
                return y_data - (a * np.log(b * x_data + c) + d)
            guess = np.array([1.0, 1.0, 1.0, 0.0])
        elif model_type == "sigmoid":
            def r(p):
                a, b, c, d = p
                return y_data - (a / (1 + _safe_exp(-b * (x_data - c))) + d)
            guess = np.array([3.0, 0.5, np.median(x_data), 0.0])
        elif model_type == "sinusoidal":
            def r(p):
                a, b, c, d = p
                return y_data - (a * np.sin(b * x_data + c) + d)
            guess = np.array([2.0, 1.0, 0.0, 0.0])
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Use a high maxfev like the reference to ensure convergence.
        params_opt, ier = leastsq(r, guess, maxfev=10000)
        
        return {"params": params_opt.tolist()}