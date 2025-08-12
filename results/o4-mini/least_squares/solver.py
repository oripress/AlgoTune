import numpy as np
from scipy.optimize import curve_fit

def _safe_exp(z):
    # Clip exponent to avoid overflow
    return np.exp(np.clip(z, -50.0, 50.0))

class Solver:
    def solve(self, problem, **kwargs):
        x = np.asarray(problem["x_data"], dtype=float)
        y = np.asarray(problem["y_data"], dtype=float)
        model = problem["model_type"]

        # Polynomial via direct polyfit
        if model == "polynomial":
            deg = problem["degree"]
            coeffs = np.polyfit(x, y, deg)
            return {"params": coeffs.tolist()}

        # Define model and initial guess
        if model == "exponential":
            def f(x, a, b, c):
                return a * _safe_exp(b * x) + c
            p0 = [1.0, 0.05, 0.0]
            bounds = (-np.inf, np.inf)

        elif model == "logarithmic":
            def f(x, a, b, c, d):
                return a * np.log(b * x + c) + d
            # ensure b*x + c > 0: enforce b>=0, c>=0
            p0 = [1.0, 1.0, 1.0, 0.0]
            lower = [-np.inf, 0.0, 0.0, -np.inf]
            upper = [ np.inf, np.inf, np.inf,  np.inf]
            bounds = (lower, upper)

        elif model == "sigmoid":
            def f(x, a, b, c, d):
                return a / (1.0 + _safe_exp(-b * (x - c))) + d
            amp = float(np.max(y) - np.min(y)) or 1.0
            p0 = [amp, 1.0, float(np.median(x)), float(np.min(y))]
            bounds = (-np.inf, np.inf)

        elif model == "sinusoidal":
            def f(x, a, b, c, d):
                return a * np.sin(b * x + c) + d
            amp = (np.max(y) - np.min(y)) / 2.0 or 1.0
            p0 = [amp, 1.0, 0.0, float(np.mean(y))]
            bounds = (-np.inf, np.inf)

        else:
            raise ValueError(f"Unknown model type: {model}")

        # Fit using curve_fit (C speed); apply bounds for problematic cases
        popt, _ = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=10000)
        return {"params": popt.tolist()}