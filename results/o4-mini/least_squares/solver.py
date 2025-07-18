import numpy as np
from scipy.optimize import leastsq
from typing import Any, Dict

def _safe_exp(z):
    """Exponentiation clipped to avoid overflow."""
    return np.exp(np.clip(z, -50.0, 50.0))

class Solver:
    def _create_residual_function(self, problem: Dict[str, Any]):
        x_data = np.asarray(problem["x_data"], dtype=float)
        y_data = np.asarray(problem["y_data"], dtype=float)
        model_type = problem["model_type"]

        if model_type == "polynomial":
            deg = int(problem["degree"])
            def r(p): return y_data - np.polyval(p, x_data)
            guess = np.ones(deg + 1, dtype=float)

        elif model_type == "exponential":
            def r(p):
                a, b, c = p
                return y_data - (a * _safe_exp(b * x_data) + c)
            guess = np.array([1.0, 0.05, 0.0], dtype=float)

        elif model_type == "logarithmic":
            def r(p):
                a, b, c, d = p
                return y_data - (a * np.log(b * x_data + c) + d)
            guess = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)

        elif model_type == "sigmoid":
            def r(p):
                a, b, c, d = p
                return y_data - (a / (1 + _safe_exp(-b * (x_data - c))) + d)
            guess = np.array([3.0, 0.5, float(np.median(x_data)), 0.0], dtype=float)

        elif model_type == "sinusoidal":
            def r(p):
                a, b, c, d = p
                return y_data - (a * np.sin(b * x_data + c) + d)
            guess = np.array([2.0, 1.0, 0.0, 0.0], dtype=float)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return r, guess

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Extract simple polynomial by direct polyfit
        model = problem["model_type"]
        if model == "polynomial":
            x = np.asarray(problem["x_data"], dtype=float)
            y = np.asarray(problem["y_data"], dtype=float)
            deg = int(problem["degree"])
            p_opt = np.polyfit(x, y, deg)
            return {"params": p_opt.tolist()}

        # For all other models, use reference residual + least squares
        residual, guess = self._create_residual_function(problem)
        # only need the optimized parameters
        p_opt = leastsq(residual, guess, maxfev=10000)
        # if full_output is False, leastsq returns only p_opt
        if isinstance(p_opt, tuple):
            p_opt = p_opt[0]
        return {"params": p_opt.tolist()}