from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np
from scipy.optimize import leastsq

class Solver:
    """Fast least squares fitter for several model families."""

    # -------------------------------
    # Utilities
    # -------------------------------
    @staticmethod
    def _safe_exp(z: np.ndarray | float) -> np.ndarray | float:
        return np.exp(np.clip(z, -50.0, 50.0))

    @staticmethod
    def _to_numpy_xy(problem: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(problem["x_data"], dtype=float)
        y = np.asarray(problem["y_data"], dtype=float)
        return x, y

    # -------------------------------
    # Core LM solver (Gauss-Newton with damping)
    # -------------------------------
    def _lm(
        self,
        r_and_jac: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        p0: np.ndarray,
        domain_ok: Callable[[np.ndarray], bool] | None = None,
        max_iter: int = 50,
    ) -> np.ndarray:
        """Levenberg–Marquardt with analytic Jacobian.

        r_and_jac(p): returns residual vector r (y - f(p)) and Jacobian of r wrt p.
        domain_ok(p): returns True if parameters are in valid domain (optional).
        """
        p = np.asarray(p0, dtype=float)
        n_params = p.size

        # Ensure initial parameters are valid if domain is constrained
        if domain_ok is not None and not domain_ok(p):
            # Try to nudge towards feasibility by small adjustments
            # If still invalid, fallback to small random perturbation around p0
            for _ in range(5):
                p_try = p + (np.random.randn(n_params) * 1e-3)
                if domain_ok(p_try):
                    p = p_try
                    break
            # If still invalid, just return p0 (best effort)
            if not domain_ok(p):
                return p

        r, J = r_and_jac(p)
        # Guard against NaNs
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(J))):
            return p

        cost = 0.5 * float(np.dot(r, r))
        # Initialize damping parameter based on Hessian approximation scale
        A = J.T @ J
        g = J.T @ r
        lam = 1e-3 * (np.trace(A) / max(n_params, 1) + 1e-12)

        # Early stopping thresholds
        gtol = 1e-8
        xtol = 1e-10
        ftol = 1e-12

        for _ in range(max_iter):
            # Check gradient sup-norm
            if np.linalg.norm(g, ord=np.inf) < gtol:
                break

            # Levenberg-Marquardt step: solve (A + lam*D) delta = -g
            diagA = np.clip(np.diag(A), 1e-12, None)
            M = A + lam * np.diag(diagA)
            try:
                delta = np.linalg.solve(M, -g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(M, -g, rcond=None)[0]

            if not np.all(np.isfinite(delta)):
                lam *= 10.0
                if lam > 1e12:
                    break
                continue

            if np.linalg.norm(delta) <= xtol * (np.linalg.norm(p) + xtol):
                break

            p_new = p + delta

            if domain_ok is not None and not domain_ok(p_new):
                lam *= 10.0
                if lam > 1e12:
                    break
                continue

            r_new, J_new = r_and_jac(p_new)
            if not (np.all(np.isfinite(r_new)) and np.all(np.isfinite(J_new))):
                lam *= 10.0
                if lam > 1e12:
                    break
                continue

            cost_new = 0.5 * float(np.dot(r_new, r_new))

            if cost_new <= cost * (1.0 - ftol) or cost_new < cost:
                # Accept
                p = p_new
                r = r_new
                J = J_new
                cost = cost_new
                A = J.T @ J
                g = J.T @ r
                lam = max(lam * 0.3, 1e-12)
            else:
                # Reject and increase damping
                lam *= 10.0
                if lam > 1e12:
                    break

        return p

    # -------------------------------
    # Initial guesses (kept for potential extensions)
    # -------------------------------
    def _guess_exponential(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        rng = y_max - y_min
        # Baseline
        c0 = y_min - 0.05 * rng
        # Guard to avoid log of non-positive
        yz = np.maximum(y - c0, 1e-8)
        lx = np.log(yz)
        # Linear regression lx ≈ ln(a) + b x
        x_mean = float(np.mean(x))
        lx_mean = float(np.mean(lx))
        x_centered = x - x_mean
        lx_centered = lx - lx_mean
        denom = float(np.dot(x_centered, x_centered)) + 1e-12
        b0 = float(np.dot(x_centered, lx_centered) / denom)
        ln_a0 = lx_mean - b0 * x_mean
        a0 = float(np.exp(np.clip(ln_a0, -50.0, 50.0)))
        # Reasonable bounds
        if not np.isfinite(a0) or a0 <= 0:
            a0 = max(rng, 1.0)
        b0 = float(np.clip(b0, -10.0, 10.0))
        return np.array([a0, b0, c0], dtype=float)

    def _guess_logarithmic(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Choose b0 positive and c0 to ensure min(b x + c) > 0
        b0 = 1.0
        # ensure strictly positive argument
        x_min = float(np.min(x))
        c0 = max(1e-3, 1.0 - b0 * x_min)
        # Estimate a0, d0 via two-point and mean matching
        # Pick far apart points in x
        i1 = int(np.argmin(x))
        i2 = int(np.argmax(x))
        x1 = float(x[i1])
        x2 = float(x[i2])
        y1 = float(y[i1])
        y2 = float(y[i2])
        l1 = np.log(b0 * x1 + c0)
        l2 = np.log(b0 * x2 + c0)
        denom = (l2 - l1)
        if abs(denom) < 1e-8:
            a0 = 1.0
        else:
            a0 = (y2 - y1) / denom
        # Offset to roughly match mean
        d0 = float(np.mean(y) - a0 * np.mean(np.log(b0 * x + c0)))
        # Clip to reasonable range
        if not np.isfinite(a0):
            a0 = 1.0
        if not np.isfinite(d0):
            d0 = 0.0
        return np.array([a0, b0, c0, d0], dtype=float)

    def _guess_sigmoid(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        d0 = y_min
        a0 = max(y_max - y_min, 1.0)
        # mid level
        mid = d0 + 0.5 * a0
        idx = int(np.argmin(np.abs(y - mid)))
        c0 = float(x[idx])
        # slope estimate near c0
        if 0 < idx < len(x) - 1:
            dx = float(x[idx + 1] - x[idx - 1])
            dy = float(y[idx + 1] - y[idx - 1])
            slope = dy / (dx if abs(dx) > 1e-12 else 1.0)
        else:
            # Fallback: global slope
            dx_tot = float(x[-1] - x[0])
            dy_tot = float(y[-1] - y[0])
            slope = dy_tot / (dx_tot if abs(dx_tot) > 1e-12 else 1.0)
        b0 = float(4.0 * slope / (a0 if a0 != 0 else 1.0))
        # Reasonable defaults if degenerate
        if not np.isfinite(b0) or abs(b0) < 1e-6:
            b0 = 0.5
        return np.array([a0, b0, c0, d0], dtype=float)

    def _guess_sinusoidal(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Offset (d)
        d0 = float(np.mean(y))
        yd = y - d0
        # Frequency estimate via FFT (assumes roughly uniform spacing)
        if len(x) > 1:
            dx = float(np.median(np.diff(x)))
            if dx <= 0 or not np.isfinite(dx):
                dx = 1.0
            Y = np.fft.rfft(yd)
            freqs = np.fft.rfftfreq(len(yd), d=dx)  # cycles per unit x
            amp = np.abs(Y)
            if amp.shape[0] > 1:
                k = 1 + int(np.argmax(amp[1:]))
                freq = float(freqs[k])
                b0 = float(2.0 * np.pi * max(freq, 1e-3))
            else:
                b0 = 1.0
        else:
            b0 = 1.0

        # Given b0, estimate a and c via linear least squares:
        s = np.sin(b0 * x)
        c = np.cos(b0 * x)
        Z = np.column_stack((s, c))
        try:
            coeffs, *_ = np.linalg.lstsq(Z, yd, rcond=None)
            As, Ac = coeffs
            a0 = float(np.hypot(As, Ac))
            c0 = float(np.arctan2(Ac, As))
        except np.linalg.LinAlgError:
            a0 = max((float(np.max(y)) - float(np.min(y))) * 0.5, 1.0)
            c0 = 0.0
        # Reasonable defaults
        if not np.isfinite(a0) or a0 == 0.0:
            a0 = max((float(np.max(y)) - float(np.min(y))) * 0.5, 1.0)
        return np.array([a0, b0, c0, d0], dtype=float)

    # -------------------------------
    # Model-specific residuals and Jacobians
    # -------------------------------
    def _res_jac_exponential(
        self, x: np.ndarray, y: np.ndarray
    ) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        def rj(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            a, b, c = p
            t = self._safe_exp(b * x)
            f = a * t + c
            r = y - f
            # Jacobian of r
            J = np.empty((x.size, 3), dtype=float)
            J[:, 0] = -t
            J[:, 1] = -a * x * t
            J[:, 2] = -1.0
            return r, J

        return rj

    def _res_jac_logarithmic(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[
        Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]], Callable[[np.ndarray], bool]
    ]:
        def domain_ok(p: np.ndarray) -> bool:
            _, b, c, _ = p
            return np.all(b * x + c > 0.0)

        def rj(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            a, b, c, d = p
            bx_c = b * x + c
            # Assume domain_ok has been checked; still guard lightly
            bx_c = np.maximum(bx_c, 1e-12)
            L = np.log(bx_c)
            f = a * L + d
            r = y - f
            inv = 1.0 / bx_c
            J = np.empty((x.size, 4), dtype=float)
            J[:, 0] = -L
            J[:, 1] = -a * x * inv
            J[:, 2] = -a * inv
            J[:, 3] = -1.0
            return r, J

        return rj, domain_ok

    def _res_jac_sigmoid(
        self, x: np.ndarray, y: np.ndarray
    ) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        def rj(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            a, b, c, d = p
            t = b * (x - c)
            s = 1.0 / (1.0 + self._safe_exp(-t))
            f = a * s + d
            r = y - f
            s1m = s * (1.0 - s)
            J = np.empty((x.size, 4), dtype=float)
            J[:, 0] = -s
            J[:, 1] = -a * s1m * (x - c)
            J[:, 2] = a * b * s1m
            J[:, 3] = -1.0
            return r, J

        return rj

    def _res_jac_sinusoidal(
        self, x: np.ndarray, y: np.ndarray
    ) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        def rj(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            a, b, c, d = p
            bx_c = b * x + c
            s = np.sin(bx_c)
            co = np.cos(bx_c)
            f = a * s + d
            r = y - f
            J = np.empty((x.size, 4), dtype=float)
            J[:, 0] = -s
            J[:, 1] = -a * co * x
            J[:, 2] = -a * co
            J[:, 3] = -1.0
            return r, J

        return rj

    # -------------------------------
    # Public API
    # -------------------------------
    def solve(self, problem: dict[str, Any], **_: Any) -> dict[str, Any]:
        x, y = self._to_numpy_xy(problem)
        model = problem["model_type"]

        if model == "polynomial":
            deg = int(problem["degree"])
            # Solve via linear least squares on Vandermonde (global optimum)
            V = np.vander(x, N=deg + 1, increasing=False)
            coeffs, *_ = np.linalg.lstsq(V, y, rcond=None)
            params = coeffs

        elif model == "exponential":
            # Use the same initial guess as the reference
            p0 = np.array([1.0, 0.05, 0.0], dtype=float)

            def r(p: np.ndarray) -> np.ndarray:
                a, b, c = p
                return y - (a * self._safe_exp(b * x) + c)

            params, _ = leastsq(r, p0, maxfev=10000)

        elif model == "logarithmic":
            p0 = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)

            def r(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y - (a * np.log(b * x + c) + d)

            params, _ = leastsq(r, p0, maxfev=10000)

        elif model == "sigmoid":
            p0 = np.array([3.0, 0.5, float(np.median(x)), 0.0], dtype=float)

            def r(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y - (a / (1 + self._safe_exp(-b * (x - c))) + d)

            params, _ = leastsq(r, p0, maxfev=10000)

        elif model == "sinusoidal":
            p0 = np.array([2.0, 1.0, 0.0, 0.0], dtype=float)

            def r(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y - (a * np.sin(b * x + c) + d)

            params, _ = leastsq(r, p0, maxfev=10000)
        else:
            raise ValueError(f"Unknown model type: {model}")

        return {"params": np.asarray(params, dtype=float).tolist()}