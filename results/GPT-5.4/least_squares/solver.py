from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.optimize import leastsq

def _safe_exp(z: np.ndarray | float) -> np.ndarray | float:
    return np.exp(np.clip(z, -50.0, 50.0))

class Solver:
    def __init__(self) -> None:
        self._success_codes = {1, 2, 3, 4}
        self._eps = 1e-12

    def _poly_solve(self, x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
        if degree == 0:
            return np.array([float(np.mean(y))], dtype=float)

        if degree == 1:
            xm = float(np.mean(x))
            ym = float(np.mean(y))
            xc = x - xm
            denom = float(np.dot(xc, xc))
            if denom <= self._eps:
                return np.array([0.0, ym], dtype=float)
            m = float(np.dot(xc, y - ym) / denom)
            b = ym - m * xm
            return np.array([m, b], dtype=float)

        if degree == 2:
            xm = float(np.mean(x))
            z = x - xm
            z2 = z * z
            s0 = float(x.size)
            s2 = float(np.sum(z2))
            if s2 > self._eps and s0 > 0.0:
                s3 = float(np.dot(z2, z))
                s4 = float(np.dot(z2, z2))
                t0 = float(np.sum(y))
                t1 = float(np.dot(z, y))
                t2 = float(np.dot(z2, y))
                den = s4 - (s3 * s3) / s2 - (s2 * s2) / s0
                if abs(den) > self._eps:
                    a2 = (t2 - (t1 * s3) / s2 - (t0 * s2) / s0) / den
                    b1 = (t1 - a2 * s3) / s2
                    c0 = (t0 - a2 * s2) / s0
                    return np.array(
                        [a2, b1 - 2.0 * a2 * xm, a2 * xm * xm - b1 * xm + c0],
                        dtype=float,
                    )
        if degree == 3:
            xm = float(np.mean(x))
            z = x - xm
            z2 = z * z
            z3 = z2 * z
            s0 = float(x.size)
            s2 = float(np.sum(z2))
            s3 = float(np.sum(z3))
            s4 = float(np.dot(z2, z2))
            s5 = float(np.dot(z2, z3))
            s6 = float(np.dot(z3, z3))
            t0 = float(np.sum(y))
            t1 = float(np.dot(z, y))
            t2 = float(np.dot(z2, y))
            t3 = float(np.dot(z3, y))
            m = np.array(
                [[s6, s5, s4, s3], [s5, s4, s3, s2], [s4, s3, s2, 0.0], [s3, s2, 0.0, s0]],
                dtype=float,
            )
            rhs = np.array([t3, t2, t1, t0], dtype=float)
            try:
                a3, b2, c1, d0 = np.linalg.solve(m, rhs)
                xm2 = xm * xm
                return np.array(
                    [
                        a3,
                        b2 - 3.0 * a3 * xm,
                        c1 - 2.0 * b2 * xm + 3.0 * a3 * xm2,
                        d0 - c1 * xm + b2 * xm2 - a3 * xm2 * xm,
                    ],
                    dtype=float,
                )
            except np.linalg.LinAlgError:
                pass

        v = np.vander(x, degree + 1)
        params, *_ = np.linalg.lstsq(v, y, rcond=None)
        return params

    def _create_residual_function(
        self, problem: dict[str, Any], x: np.ndarray, y: np.ndarray
    ) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
        model_type = problem["model_type"]

        if model_type == "exponential":

            def residual(p: np.ndarray) -> np.ndarray:
                a, b, c = p
                return y - (a * _safe_exp(b * x) + c)

            guess = np.array([1.0, 0.05, 0.0], dtype=float)

        elif model_type == "logarithmic":

            def residual(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y - (a * np.log(b * x + c) + d)

            guess = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)

        elif model_type == "sigmoid":
            x_med = float(np.median(x))

            def residual(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y - (a / (1.0 + _safe_exp(-b * (x - c))) + d)

            guess = np.array([3.0, 0.5, x_med, 0.0], dtype=float)

        elif model_type == "sinusoidal":

            def residual(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y - (a * np.sin(b * x + c) + d)

            guess = np.array([2.0, 1.0, 0.0, 0.0], dtype=float)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return residual, guess

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        x = np.asarray(problem["x_data"], dtype=float)
        y = np.asarray(problem["y_data"], dtype=float)
        model_type = problem["model_type"]

        if model_type == "polynomial":
            params = self._poly_solve(x, y, int(problem["degree"]))
            return {"params": params.tolist()}

        residual, guess = self._create_residual_function(problem, x, y)
        params_opt = leastsq(residual, guess, full_output=False, maxfev=10000)[0]
        return {"params": params_opt.tolist()}