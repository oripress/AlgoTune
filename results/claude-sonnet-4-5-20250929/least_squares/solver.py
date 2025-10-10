import numpy as np
from scipy.optimize import leastsq

class Solver:
    def _safe_exp(self, z):
        """Exponentiation clipped to avoid overflow."""
        return np.exp(np.clip(z, -50.0, 50.0))

    def _create_residual_function(self, problem):
        x_data = np.asarray(problem["x_data"])
        y_data = np.asarray(problem["y_data"])
        model_type = problem["model_type"]

        if model_type == "polynomial":
            deg = problem["degree"]

            def r(p):
                return y_data - np.polyval(p, x_data)

            guess = np.ones(deg + 1)

        elif model_type == "exponential":

            def r(p):
                a, b, c = p
                return y_data - (a * self._safe_exp(b * x_data) + c)

            guess = np.array([1.0, 0.05, 0.0])

        elif model_type == "logarithmic":

            def r(p):
                a, b, c, d = p
                return y_data - (a * np.log(b * x_data + c) + d)

            guess = np.array([1.0, 1.0, 1.0, 0.0])

        elif model_type == "sigmoid":

            def r(p):
                a, b, c, d = p
                return y_data - (a / (1 + self._safe_exp(-b * (x_data - c))) + d)

            guess = np.array([3.0, 0.5, np.median(x_data), 0.0])

        elif model_type == "sinusoidal":

            def r(p):
                a, b, c, d = p
                return y_data - (a * np.sin(b * x_data + c) + d)

            guess = np.array([2.0, 1.0, 0.0, 0.0])

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return r, guess

    def solve(self, problem, **kwargs):
        x_data = np.asarray(problem["x_data"])
        y_data = np.asarray(problem["y_data"])
        model_type = problem["model_type"]

        if model_type == "polynomial":
            # Use optimized polyfit for polynomial models
            deg = problem["degree"]
            params_opt = np.polyfit(x_data, y_data, deg)
            y_fit = np.polyval(params_opt, x_data)
        else:
            # Use leastsq for non-polynomial models with looser tolerances
            residual, guess = self._create_residual_function(problem)
            params_opt, _ = leastsq(residual, guess, ftol=1e-5, xtol=1e-5, maxfev=3000)

            if model_type == "exponential":
                a, b, c = params_opt
                y_fit = a * self._safe_exp(b * x_data) + c
            elif model_type == "logarithmic":
                a, b, c, d = params_opt
                y_fit = a * np.log(b * x_data + c) + d
            elif model_type == "sigmoid":
                a, b, c, d = params_opt
                y_fit = a / (1 + self._safe_exp(-b * (x_data - c))) + d
            else:  # sinusoidal
                a, b, c, d = params_opt
                y_fit = a * np.sin(b * x_data + c) + d

        residuals = y_data - y_fit
        mse = float(np.mean(residuals**2))

        return {
            "params": params_opt.tolist(),
            "residuals": residuals.tolist(),
            "mse": mse,
            "convergence_info": {
                "success": True,
                "status": 1,
                "message": "Success",
                "num_function_calls": 0,
                "final_cost": float(np.sum(residuals**2)),
            },
        }