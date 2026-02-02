from __future__ import annotations

from typing import Any, Dict, Literal

import numpy as np

try:
    # SciPy is available in the environment (used by the reference).
    from scipy.integrate import tanhsinh
    from scipy.special import wright_bessel
except Exception:  # pragma: no cover
    tanhsinh = None  # type: ignore[assignment]
    wright_bessel = None  # type: ignore[assignment]

Mode = Literal["xab", "abx", "numeric"]

class Solver:
    """
    Fast solver for elementwise integrals of Wright's Bessel function.

    We exploit the series identity (when integrating w.r.t. the series variable x):
        d/dx Phi(a, beta; x) = Phi(a, beta + a; x)
    hence:
        ∫ Phi(a,b;x) dx = Phi(a, b-a; x) + C

    However, correctness depends on SciPy's positional argument order for
    `scipy.special.wright_bessel` relative to tanhsinh calling f(x, *args).
    We calibrate once in __init__ against a single tanhsinh integral and then
    use the matching fast closed form; otherwise we fall back to tanhsinh.
    """

    def __init__(self) -> None:
        self._wb = wright_bessel
        self._tanhsinh = tanhsinh
        self._mode: Mode = "numeric"

        if self._wb is None or self._tanhsinh is None:  # pragma: no cover
            return

        wb = self._wb
        th = self._tanhsinh

        # Calibrate on a fixed, well-behaved point (positive interval, typical a,b).
        a0 = 1.3
        b0 = 2.1
        l0 = 0.2
        u0 = 0.7

        try:
            ref = float(th(wb, l0, u0, args=(a0, b0)).integral)
        except Exception:  # pragma: no cover
            self._mode = "numeric"
            return

        bb0 = b0 - a0

        # Candidate 1: wright_bessel(x, a, b) -> Phi(a,b;x)
        try:
            cand_xab = float(wb(u0, a0, bb0) - wb(l0, a0, bb0))
        except Exception:
            cand_xab = np.nan

        # Candidate 2: wright_bessel(a, b, x) -> Phi(a,b;x)
        try:
            cand_abx = float(wb(a0, bb0, u0) - wb(a0, bb0, l0))
        except Exception:
            cand_abx = np.nan

        def relerr(x: float, y: float) -> float:
            if not (np.isfinite(x) and np.isfinite(y)):
                return np.inf
            denom = max(1.0, abs(y))
            return abs(x - y) / denom

        ex = relerr(cand_xab, ref)
        ea = relerr(cand_abx, ref)

        # Pick the formula that best matches tanhsinh.
        if ex <= ea:
            self._mode = "xab"
        else:
            self._mode = "abx"

        # If neither is even close, use numeric mode.
        if min(ex, ea) > 1e-7:
            self._mode = "numeric"

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        wb = self._wb
        if wb is None:  # pragma: no cover
            raise RuntimeError("scipy.special.wright_bessel is required but not available")

        # Debug hook (never used by the evaluator).
        if problem.get("_debug_doc", False):
            doc = getattr(wb, "__doc__", "") or ""
            sig = getattr(wb, "__text_signature__", "") or ""
            print("wright_bessel.__text_signature__:", sig)
            print("wright_bessel.__doc__ (head):", doc[:300].replace("\n", " "))

        a = np.asarray(problem["a"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        lower = np.asarray(problem["lower"], dtype=np.float64)
        upper = np.asarray(problem["upper"], dtype=np.float64)

        mode: Mode = self._mode

        if mode != "numeric":
            bb = b - a
            if mode == "xab":
                res = wb(upper, a, bb) - wb(lower, a, bb)
            else:  # "abx"
                res = wb(a, bb, upper) - wb(a, bb, lower)

            if np.all(np.isfinite(res)):
                return {"result": res.tolist()}
            # else: fall through to numeric per-element fallback

        th = self._tanhsinh
        if th is None:  # pragma: no cover
            raise RuntimeError("scipy.integrate.tanhsinh is required but not available")

        # Vectorized tanhsinh call (matches the reference; much faster than looping).
        tres = th(wb, lower, upper, args=(a, b))
        return {"result": np.asarray(tres.integral, dtype=np.float64).tolist()}