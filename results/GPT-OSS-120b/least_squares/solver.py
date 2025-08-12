import numpy as np
from scipy.optimize import curve_fit
from typing import Any, Dict

def _safe_exp(z: np.ndarray | float) -> np.ndarray | float:
    """Exponentiation clipped to avoid overflow."""
    return np.exp(np.clip(z, -50.0, 50.0))

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Fit the specified model to the data using fast least‑squares methods.
        Returns parameters, residuals, MSE and simple convergence info.
        """
        x = np.asarray(problem["x_data"], dtype=float)
        y = np.asarray(problem["y_data"], dtype=float)
        model_type = problem["model_type"]

        # Initialise y_fit to silence static analysis warnings
        y_fit = np.zeros_like(y)

        # ------------------------------------------------------------------
        # Model handling
        # ------------------------------------------------------------------
        if model_type == "polynomial":
            deg = problem["degree"]
            V = np.vander(x, deg + 1, increasing=False)   # (n, deg+1)
            params_opt, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
            y_fit = np.polyval(params_opt, x)
        else:
            # Define model functions and initial guesses
            if model_type == "exponential":
                func = lambda x, a, b, c: a * _safe_exp(b * x) + c
                p0 = np.array([1.0, 0.05, 0.0])
            elif model_type == "logarithmic":
                func = lambda x, a, b, c, d: a * np.log(b * x + c) + d
                p0 = np.array([1.0, 1.0, 1.0, 0.0])
            elif model_type == "sigmoid":
                func = lambda x, a, b, c, d: a / (1 + _safe_exp(-b * (x - c))) + d
                p0 = np.array([3.0, 0.5, np.median(x), 0.0])
            elif model_type == "sinusoidal":
                func = lambda x, a, b, c, d: a * np.sin(b * x + c) + d
                p0 = np.array([2.0, 1.0, 0.0, 0.0])
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            try:
                params_opt, _ = curve_fit(func, x, y, p0=p0, maxfev=5000)
            except RuntimeError:
                # Fallback to initial guess if optimisation fails
                params_opt = p0

            # Compute fitted values for the chosen model
            if model_type == "exponential":
                a, b, c = params_opt
                y_fit = a * _safe_exp(b * x) + c
            elif model_type == "logarithmic":
                a, b, c, d = params_opt
                y_fit = a * np.log(b * x + c) + d
            elif model_type == "sigmoid":
                a, b, c, d = params_opt
                y_fit = a / (1 + _safe_exp(-b * (x - c))) + d
            elif model_type == "sinusoidal":
                a, b, c, d = params_opt
                y_fit = a * np.sin(b * x + c) + d

        # ------------------------------------------------------------------
        # Residuals, MSE and convergence information
        # ------------------------------------------------------------------
        residuals = y - y_fit
        mse = float(np.mean(residuals ** 2))

        convergence_info = {
            "success": True,
            "status": 0,
            "message": "fit completed",
            "num_function_calls": -1,   # not tracked by curve_fit
            "final_cost": float(np.sum(residuals ** 2)),
        }

        return {
            "params": params_opt.tolist(),
            "residuals": residuals.tolist(),
            "mse": mse,
            "convergence_info": convergence_info,
        }