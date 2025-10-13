from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.optimize import leastsq

def _safe_exp(z: np.ndarray | float) -> np.ndarray | float:
    # Exponentiation clipped to avoid overflow, matching reference behavior
    return np.exp(np.clip(z, -50.0, 50.0))

class Solver:
    def _create_residual_function(
        self, problem: dict[str, Any]
    ) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
        x_data = np.asarray(problem["x_data"], dtype=float)
        y_data = np.asarray(problem["y_data"], dtype=float)
        model_type = problem["model_type"]

        if model_type == "polynomial":
            deg = int(problem["degree"])

            def r(p: np.ndarray) -> np.ndarray:
                return y_data - np.polyval(p, x_data)

            guess = np.ones(deg + 1, dtype=float)

        elif model_type == "exponential":

            def r(p: np.ndarray) -> np.ndarray:
                a, b, c = p
                return y_data - (a * _safe_exp(b * x_data) + c)

            guess = np.array([1.0, 0.05, 0.0], dtype=float)

        elif model_type == "logarithmic":

            def r(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y_data - (a * np.log(b * x_data + c) + d)

            guess = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)

        elif model_type == "sigmoid":

            def r(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y_data - (a / (1 + _safe_exp(-b * (x_data - c))) + d)

            guess = np.array([3.0, 0.5, np.median(x_data), 0.0], dtype=float)

        elif model_type == "sinusoidal":

            def r(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y_data - (a * np.sin(b * x_data + c) + d)

            guess = np.array([2.0, 1.0, 0.0, 0.0], dtype=float)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return r, guess

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        model_type = problem["model_type"]
        x_data = np.asarray(problem["x_data"], dtype=float)
        y_data = np.asarray(problem["y_data"], dtype=float)

        # Fast path for polynomial: closed-form via linear least squares
        if model_type == "polynomial":
            deg = int(problem["degree"])
            # Use rcond=None for NumPy >=1.14 default behavior
            params = np.polyfit(x_data, y_data, deg)
            return {"params": params.tolist()}

        # Nonlinear models: use Levenberg–Marquardt least squares like reference
        residual, guess = self._create_residual_function(problem)
        params_opt, _cov_x, _info, _mesg, _ier = leastsq(
            residual, guess, full_output=True, maxfev=10000
        )

        return {"params": params_opt.tolist()}