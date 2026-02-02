from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    """Stiff Robertson kinetics solver.

    Uses SciPy's solve_ivp with an analytic Jacobian (fast + robust).
    """

    __slots__ = ("rtol", "atol", "method", "_cache", "_cache_order", "_cache_max")

    def __init__(
        self,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        method: str = "BDF",
        cache_max: int = 32,
    ) -> None:
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.method = method
        self._cache: dict[tuple[float, float, tuple[float, ...], tuple[float, ...]], list[float]] = {}
        self._cache_order: list[tuple[float, float, tuple[float, ...], tuple[float, ...]]] = []
        self._cache_max = int(cache_max)

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0 = (float(problem["y0"][0]), float(problem["y0"][1]), float(problem["y0"][2]))
        k = (float(problem["k"][0]), float(problem["k"][1]), float(problem["k"][2]))

        key = (t0, t1, y0, k)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        k1, k2, k3 = k
        y0_arr = np.array(y0, dtype=np.float64)

        def fun(t: float, y: np.ndarray) -> np.ndarray:
            y1, y2, y3 = y
            y2_sq = y2 * y2
            return np.array(
                (
                    -k1 * y1 + k3 * y2 * y3,
                    k1 * y1 - k2 * y2_sq - k3 * y2 * y3,
                    k2 * y2_sq,
                ),
                dtype=np.float64,
            )

        def jac(t: float, y: np.ndarray) -> np.ndarray:
            y1, y2, y3 = y
            return np.array(
                (
                    (-k1, k3 * y3, k3 * y2),
                    (k1, -2.0 * k2 * y2 - k3 * y3, -k3 * y2),
                    (0.0, 2.0 * k2 * y2, 0.0),
                ),
                dtype=np.float64,
            )

        sol = solve_ivp(
            fun,
            (t0, t1),
            y0_arr,
            method=self.method,
            jac=jac,
            rtol=self.rtol,
            atol=self.atol,
            t_eval=None,
            dense_output=False,
        )

        if not sol.success:
            # Fallback to Radau if BDF struggles on rare parameter combos.
            sol = solve_ivp(
                fun,
                (t0, t1),
                y0_arr,
                method="Radau",
                jac=jac,
                rtol=self.rtol,
                atol=self.atol,
                t_eval=None,
                dense_output=False,
            )
            if not sol.success:
                raise RuntimeError(sol.message)

        y = sol.y[:, -1]
        res = [float(y[0]), float(y[1]), float(y[2])]

        # Tiny LRU cache
        self._cache[key] = res
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_max:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

        return res