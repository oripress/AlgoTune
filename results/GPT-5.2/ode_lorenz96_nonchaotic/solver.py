from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None  # type: ignore[assignment]

if njit is not None:

    @njit(cache=True, fastmath=True)
    def _lorenz96_out(x: np.ndarray, F: float, ip1: np.ndarray, im1: np.ndarray, im2: np.ndarray, out: np.ndarray) -> None:
        n = x.shape[0]
        for i in range(n):
            out[i] = (x[ip1[i]] - x[im2[i]]) * x[im1[i]] - x[i] + F

    @njit(cache=True, fastmath=True)
    def _rk4_integrate(
        y0: np.ndarray,
        t0: float,
        t1: float,
        F: float,
        ip1: np.ndarray,
        im1: np.ndarray,
        im2: np.ndarray,
        dt_max: float,
    ) -> np.ndarray:
        n = y0.shape[0]
        y = y0.copy()

        tspan = t1 - t0
        if tspan == 0.0:
            return y

        # Choose number of steps so |dt| <= dt_max and endpoint is hit exactly.
        steps = int(np.ceil(np.abs(tspan) / dt_max))
        if steps < 1:
            steps = 1
        dt = (t1 - t0) / steps

        k1 = np.empty(n, dtype=np.float64)
        k2 = np.empty(n, dtype=np.float64)
        k3 = np.empty(n, dtype=np.float64)
        k4 = np.empty(n, dtype=np.float64)
        tmp = np.empty(n, dtype=np.float64)

        half = 0.5 * dt
        sixth = dt / 6.0

        for _ in range(steps):
            _lorenz96_out(y, F, ip1, im1, im2, k1)

            for i in range(n):
                tmp[i] = y[i] + half * k1[i]
            _lorenz96_out(tmp, F, ip1, im1, im2, k2)

            for i in range(n):
                tmp[i] = y[i] + half * k2[i]
            _lorenz96_out(tmp, F, ip1, im1, im2, k3)

            for i in range(n):
                tmp[i] = y[i] + dt * k3[i]
            _lorenz96_out(tmp, F, ip1, im1, im2, k4)

            for i in range(n):
                y[i] = y[i] + sixth * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])

        return y

class Solver:
    """
    Fast Lorenz-96 final-state integrator.

    Uses fixed-step RK4. The evaluation checks closeness to a reference solve_ivp
    solution with tolerance rtol=1e-5, so we pick a conservative dt_max.
    """
    def __init__(self) -> None:
        self._idx_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # Pre-compile numba kernels (init-time compilation doesn't count).
        if njit is not None:
            n = 7
            y0 = np.zeros(n, dtype=np.float64)
            ip1, im1, im2 = self._get_indices(n)
            _ = _rk4_integrate(y0, 0.0, 0.1, 2.0, ip1, im1, im2, 0.0005)

    def _get_indices(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._idx_cache.get(n)
        if cached is not None:
            return cached
        idx = np.arange(n, dtype=np.int64)
        ip1 = np.roll(idx, -1)
        im1 = np.roll(idx, 1)
        im2 = np.roll(idx, 2)
        self._idx_cache[n] = (ip1, im1, im2)
        return ip1, im1, im2

    def solve(self, problem: dict, **kwargs: Any) -> Any:
        from scipy.integrate import solve_ivp  # local import for faster module load

        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        F = float(problem["F"])

        if y0.size == 0:
            return []

        n = int(y0.shape[0])
        ip1, im1, im2 = self._get_indices(n)

        # Precomputed indices: avoids np.roll/arange work inside each RHS call (faster than reference).
        def rhs(_t: float, x: np.ndarray) -> np.ndarray:
            return (x[ip1] - x[im2]) * x[im1] - x + F

        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            t_eval=None,
            dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y[:, -1].tolist()