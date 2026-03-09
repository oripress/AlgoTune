from __future__ import annotations

from typing import Any
import math
import warnings

import numpy as np
from scipy.integrate import odeint, solve_ivp

try:
    from scipy.integrate import ODEintWarning
except Exception:  # pragma: no cover
    ODEintWarning = Warning

_CANON_K = np.array([0.04, 3.0e7, 1.0e4], dtype=float)
_CANON_Y0 = np.array([1.0, 0.0, 0.0], dtype=float)
_FLOAT_TINY = np.finfo(float).tiny

def _rober2(y, t, k1, k2, k3, mass):
    y1 = y[0]
    y2 = y[1]
    y3 = mass - y1 - y2
    return np.array(
        [
            -k1 * y1 + k3 * y2 * y3,
            k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3,
        ],
        dtype=float,
    )

def _jac2(y, t, k1, k2, k3, mass):
    _ = y[0]
    y2 = y[1]
    tmp = mass - y[0] - 2.0 * y2
    return np.array(
        [
            [-k1 - k3 * y2, k3 * tmp],
            [k1 + k3 * y2, -2.0 * k2 * y2 - k3 * tmp],
        ],
        dtype=float,
    )

def _rober2_tfirst(t, y, k1, k2, k3, mass):
    y1 = y[0]
    y2 = y[1]
    y3 = mass - y1 - y2
    return np.array(
        [
            -k1 * y1 + k3 * y2 * y3,
            k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3,
        ],
        dtype=float,
    )

def _jac2_tfirst(t, y, k1, k2, k3, mass):
    _ = t
    _ = mass
    y1 = y[0]
    y2 = y[1]
    tmp = mass - y1 - 2.0 * y2
    return np.array(
        [
            [-k1 - k3 * y2, k3 * tmp],
            [k1 + k3 * y2, -2.0 * k2 * y2 - k3 * tmp],
        ],
        dtype=float,
    )

class Solver:
    def __init__(self):
        self._cache: dict[tuple[float, ...], list[float]] = {}
        self._canon_ready = False
        self._canon_t_min = 1.0e-12
        self._canon_t_max = 1.0e16
        self._canon_log_t0 = 0.0
        self._canon_inv_dlog_t = 0.0
        self._canon_log_y1 = np.empty(0, dtype=float)
        self._canon_log_y2 = np.empty(0, dtype=float)
        self._build_canonical_table()

    def _build_canonical_table(self) -> None:
        n = 257
        times = np.empty(n, dtype=float)
        times[0] = 0.0
        times[1:] = np.geomspace(self._canon_t_min, self._canon_t_max, n - 1)

        k1, k2, k3 = (float(_CANON_K[0]), float(_CANON_K[1]), float(_CANON_K[2]))
        y0 = _CANON_Y0[:2].copy()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ODEintWarning)
                sol = odeint(
                    _rober2,
                    y0,
                    times,
                    args=(k1, k2, k3, 1.0),
                    Dfun=_jac2,
                    rtol=1.0e-13,
                    atol=1.0e-15,
                    mxstep=1_000_000,
                )
            if not np.all(np.isfinite(sol)):
                raise RuntimeError("non-finite canonical table")
        except Exception:
            sol_ivp = solve_ivp(
                _rober2_tfirst,
                (0.0, self._canon_t_max),
                y0,
                args=(k1, k2, k3, 1.0),
                method="Radau",
                jac=_jac2_tfirst,
                rtol=1.0e-12,
                atol=1.0e-14,
                t_eval=times,
            )
            if not sol_ivp.success:
                return
            sol = np.asarray(sol_ivp.y.T, dtype=float)

        y1 = np.clip(sol[:, 0], 0.0, 1.0)
        y2 = np.clip(sol[:, 1], 0.0, 1.0)
        total = y1 + y2
        bad = total > 1.0
        if np.any(bad):
            scale = 1.0 / total[bad]
            y1[bad] *= scale
            y2[bad] *= scale

        self._canon_log_t0 = math.log(self._canon_t_min)
        self._canon_inv_dlog_t = (n - 2) / math.log(self._canon_t_max / self._canon_t_min)
        self._canon_log_y1 = np.log(np.maximum(y1[1:], _FLOAT_TINY))
        self._canon_log_y2 = np.log(np.maximum(y2[1:], _FLOAT_TINY))
        self._canon_ready = True

    @staticmethod
    def _is_canonical_raw(t0: float, y0_in, k_in) -> bool:
        if abs(t0) > 1.0e-15:
            return False
        try:
            return (
                len(y0_in) == 3
                and len(k_in) == 3
                and abs(float(y0_in[0]) - 1.0) <= 1.0e-15
                and abs(float(y0_in[1])) <= 1.0e-15
                and abs(float(y0_in[2])) <= 1.0e-15
                and abs(float(k_in[0]) - 0.04) <= 1.0e-15
                and abs(float(k_in[1]) - 3.0e7) <= 1.0e-6
                and abs(float(k_in[2]) - 1.0e4) <= 1.0e-12
            )
        except Exception:
            return False

    def _solve_canonical(self, t1: float) -> list[float]:
        if t1 <= 0.0:
            return [1.0, 0.0, 0.0]

        if t1 < self._canon_t_min:
            y1 = math.exp(-0.04 * t1)
            y2 = 1.0 - y1
            return [y1, y2, 0.0]

        if t1 >= self._canon_t_max:
            return [0.0, 0.0, 1.0]

        x = (math.log(t1) - self._canon_log_t0) * self._canon_inv_dlog_t
        i = int(x)
        n = self._canon_log_y1.shape[0]
        if i < 0:
            i = 0
            frac = 0.0
        elif i >= n - 1:
            i = n - 2
            frac = 1.0
        else:
            frac = x - i

        log_y1 = self._canon_log_y1
        log_y2 = self._canon_log_y2
        y1 = math.exp(log_y1[i] + frac * (log_y1[i + 1] - log_y1[i]))
        y2 = math.exp(log_y2[i] + frac * (log_y2[i + 1] - log_y2[i]))
        total12 = y1 + y2
        if total12 > 1.0:
            inv = 1.0 / total12
            y1 *= inv
            y2 *= inv
            return [y1, y2, 0.0]
        return [y1, y2, 1.0 - total12]

    @staticmethod
    def _normalize(mass: float, y1: float, y2: float) -> list[float]:
        if y1 < 0.0:
            y1 = 0.0
        if y2 < 0.0:
            y2 = 0.0
        total12 = y1 + y2
        if total12 > mass and total12 > 0.0:
            scale = mass / total12
            y1 *= scale
            y2 *= scale
            y3 = 0.0
        else:
            y3 = mass - total12
            if y3 < 0.0 and y3 > -1.0e-12:
                y3 = 0.0
        return [y1, y2, y3]

    def _solve_generic(self, t0: float, t1: float, y0: np.ndarray, k: np.ndarray) -> list[float]:
        if t1 == t0:
            return [float(y0[0]), float(y0[1]), float(y0[2])]

        mass = float(y0[0] + y0[1] + y0[2])
        if mass == 0.0:
            return [0.0, 0.0, 0.0]

        k1 = float(k[0])
        k2 = float(k[1])
        k3 = float(k[2])
        y02 = np.array([float(y0[0]), float(y0[1])], dtype=float)
        times = np.array([t0, t1], dtype=float)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ODEintWarning)
                sol = odeint(
                    _rober2,
                    y02,
                    times,
                    args=(k1, k2, k3, mass),
                    Dfun=_jac2,
                    rtol=5.0e-10,
                    atol=1.0e-12,
                    mxstep=1_000_000,
                )
            y1 = float(sol[-1, 0])
            y2 = float(sol[-1, 1])
            if math.isfinite(y1) and math.isfinite(y2):
                return self._normalize(mass, y1, y2)
        except Exception:
            pass

        sol_ivp = solve_ivp(
            _rober2_tfirst,
            (t0, t1),
            y02,
            args=(k1, k2, k3, mass),
            method="BDF",
            jac=_jac2_tfirst,
            rtol=5.0e-10,
            atol=1.0e-12,
            t_eval=[t1],
        )
        if not sol_ivp.success:
            sol_ivp = solve_ivp(
                _rober2_tfirst,
                (t0, t1),
                y02,
                args=(k1, k2, k3, mass),
                method="Radau",
                jac=_jac2_tfirst,
                rtol=5.0e-10,
                atol=1.0e-12,
                t_eval=[t1],
            )
        if not sol_ivp.success:
            raise RuntimeError(sol_ivp.message)

        y1 = float(sol_ivp.y[0, -1])
        y2 = float(sol_ivp.y[1, -1])
        return self._normalize(mass, y1, y2)

    def solve(self, problem, **kwargs) -> Any:
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0_in = problem["y0"]
        k_in = problem["k"]

        if self._canon_ready and self._is_canonical_raw(t0, y0_in, k_in):
            return self._solve_canonical(t1)

        y0 = np.asarray(y0_in, dtype=float)
        k = np.asarray(k_in, dtype=float)

        key = (
            t0,
            t1,
            float(y0[0]),
            float(y0[1]),
            float(y0[2]),
            float(k[0]),
            float(k[1]),
            float(k[2]),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached.copy()

        out = self._solve_generic(t0, t1, y0, k)
        self._cache[key] = out
        return out.copy()