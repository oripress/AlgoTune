from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import leastsq

def _safe_exp(z: np.ndarray) -> np.ndarray:
    """Exponentiation clipped to avoid overflow; matches reference behavior."""
    return np.exp(np.clip(z, -50.0, 50.0))

@dataclass
class Solver:
    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        x = np.asarray(problem["x_data"], dtype=float)
        y = np.asarray(problem["y_data"], dtype=float)
        model_type = problem["model_type"]

        if x.size == 0:
            return {"params": []}

        if model_type == "polynomial":
            deg = int(problem["degree"])
            params = np.polyfit(x, y, deg)  # descending powers, np.polyval order
            return {"params": params.tolist()}

        if model_type == "exponential":

            def resid_exp(p: np.ndarray) -> np.ndarray:
                a, b, c = p
                return y - (a * _safe_exp(b * x) + c)

            def jac_exp(p: np.ndarray) -> np.ndarray:
                a, b, _c = p
                z = b * x
                # Respect clipping: derivative is 0 if z is outside clip range.
                mask = (z > -50.0) & (z < 50.0)
                ebx = np.exp(np.clip(z, -50.0, 50.0))
                J = np.empty((x.size, 3), dtype=float)
                # residual r = y - f => dr/dp = -df/dp
                J[:, 0] = -ebx
                J[:, 1] = -(a * x * ebx * mask)
                J[:, 2] = -1.0
                return J

            guess = np.array([1.0, 0.05, 0.0], dtype=float)
            params, _ = leastsq(
                resid_exp, guess, Dfun=jac_exp, col_deriv=0, maxfev=10_000
            )
            return {"params": params.tolist()}

        if model_type == "logarithmic":
            # Keep residual exactly as reference (no clipping); numeric Jacobian is safer.
            def resid_log(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y - (a * np.log(b * x + c) + d)

            guess = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)
            params, _ = leastsq(resid_log, guess, maxfev=10_000)
            return {"params": params.tolist()}

        if model_type == "sigmoid":

            def resid_sig(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y - (a / (1.0 + _safe_exp(-b * (x - c))) + d)

            def jac_sig(p: np.ndarray) -> np.ndarray:
                a, b, c, _d = p
                t = b * (x - c)
                # safe_exp clips -t; derivative is 0 if -t clipped => t outside [-50,50]
                mask = (t > -50.0) & (t < 50.0)
                s = 1.0 / (1.0 + _safe_exp(-t))
                ds = s * (1.0 - s) * mask
                J = np.empty((x.size, 4), dtype=float)
                # f = a*s + d
                J[:, 0] = -s
                J[:, 1] = -(a * ds * (x - c))
                J[:, 2] = a * ds * b
                J[:, 3] = -1.0
                return J

            guess = np.array([3.0, 0.5, float(np.median(x)), 0.0], dtype=float)
            params, _ = leastsq(
                resid_sig, guess, Dfun=jac_sig, col_deriv=0, maxfev=10_000
            )
            return {"params": params.tolist()}

        if model_type == "sinusoidal":

            def resid_sin(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y - (a * np.sin(b * x + c) + d)

            def jac_sin(p: np.ndarray) -> np.ndarray:
                a, b, c, _d = p
                u = b * x + c
                su = np.sin(u)
                cu = np.cos(u)
                J = np.empty((x.size, 4), dtype=float)
                J[:, 0] = -su
                J[:, 1] = -(a * x * cu)
                J[:, 2] = -(a * cu)
                J[:, 3] = -1.0
                return J

            guess = np.array([2.0, 1.0, 0.0, 0.0], dtype=float)
            params, _ = leastsq(
                resid_sin, guess, Dfun=jac_sin, col_deriv=0, maxfev=10_000
            )
            return {"params": params.tolist()}

        raise ValueError(f"Unknown model type: {model_type}")