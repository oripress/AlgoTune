from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None  # type: ignore[assignment]

if njit is not None:

    @njit(cache=False, fastmath=True)
    def _water_level_sorted(a_sorted: np.ndarray, p_total: float) -> float:
        """Water level w for ascending-sorted alpha."""
        pref = 0.0
        n = a_sorted.size
        for i in range(n):
            pref += a_sorted[i]
            w = (p_total + pref) / (i + 1.0)
            if i == n - 1 or w <= a_sorted[i + 1]:
                return w
        return w  # pragma: no cover

    @njit(cache=False, fastmath=True)
    def _x_scaled_and_capacity(alpha: np.ndarray, w: float, p_total: float) -> tuple[np.ndarray, float]:
        """
        Compute x_i = max(0, w-alpha_i) then scale by gamma = p_total/sum(x),
        and compute Capacity = sum(log(alpha_i + x_i_scaled)).

        Implemented as tight loops to avoid large temporaries from numpy ufuncs.
        """
        n = alpha.size
        x = np.empty(n, dtype=np.float64)

        s = 0.0
        for i in range(n):
            v = w - alpha[i]
            if v > 0.0:
                x[i] = v
                s += v
            else:
                x[i] = 0.0

        if s <= 0.0:
            # Defensive fallback; should not happen for valid instances.
            val = p_total / n
            cap = 0.0
            for i in range(n):
                x[i] = val
                cap += np.log(alpha[i] + val)
            return x, cap

        gamma = p_total / s

        cap = 0.0
        for i in range(n):
            xi = x[i] * gamma
            x[i] = xi
            cap += np.log(alpha[i] + xi)

        return x, cap

else:
    _water_level_sorted = None  # type: ignore[assignment]
    _x_scaled_and_capacity = None  # type: ignore[assignment]

class Solver:
    def __init__(self) -> None:
        # Compile numba once (init time not counted).
        if _water_level_sorted is not None:
            _water_level_sorted(np.array([1.0, 2.0, 3.0], dtype=np.float64), 1.0)
        if _x_scaled_and_capacity is not None:
            alpha = np.array([0.5, 1.0, 2.0], dtype=np.float64)
            _x_scaled_and_capacity(alpha, 2.0, 1.0)

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        alpha = np.asarray(problem["alpha"], dtype=np.float64)
        p_total = float(problem["P_total"])
        n = int(alpha.size)

        if n == 0 or p_total <= 0.0 or (not np.isfinite(p_total)):
            return {"x": np.full(n, np.nan), "Capacity": float("nan")}

        if n == 1:
            a0 = float(alpha[0])
            if (not np.isfinite(a0)) or a0 <= 0.0:
                return {"x": np.full(1, np.nan), "Capacity": float("nan")}
            x0 = float(p_total)
            return {"x": np.array([x0], dtype=np.float64), "Capacity": float(np.log(a0 + x0))}

        # Sort ascending alpha (needed to find water level).
        a_sorted = np.sort(alpha)

        # Match reference invalid handling: any non-positive alpha => NaN.
        # (NaNs will not pass this check since comparisons are False.)
        if not (a_sorted[0] > 0.0):
            return {"x": np.full(n, np.nan), "Capacity": float("nan")}

        if _water_level_sorted is not None:
            w = float(_water_level_sorted(a_sorted, p_total))
        else:
            pref = 0.0
            w = 0.0
            for i in range(n):
                pref += float(a_sorted[i])
                w = (p_total + pref) / (i + 1.0)
                if i == n - 1 or w <= float(a_sorted[i + 1]):
                    break

        if _x_scaled_and_capacity is not None:
            x, cap = _x_scaled_and_capacity(alpha, w, p_total)
            return {"x": x, "Capacity": float(cap)}

        # Fallback path without numba.
        x = w - alpha
        np.maximum(x, 0.0, out=x)
        s = float(x.sum())
        if s > 0.0:
            x *= (p_total / s)
        cap = float(np.log(alpha + x).sum())
        return {"x": x, "Capacity": cap}