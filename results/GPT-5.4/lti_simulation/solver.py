from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit
from scipy import signal
from scipy.linalg import expm

_lsim = signal.lsim
_tf2ss = signal.tf2ss

@njit(cache=False)
def _simulate_ss_numba(
    ad: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    c: np.ndarray,
    d: float,
    u: np.ndarray,
) -> np.ndarray:
    n_steps = u.shape[0]
    n_state = c.shape[0]
    y = np.empty(n_steps, dtype=np.float64)

    if n_state == 0:
        for i in range(n_steps):
            y[i] = d * u[i]
        return y

    x = np.zeros(n_state, dtype=np.float64)
    x_new = np.empty(n_state, dtype=np.float64)

    for i in range(n_steps - 1):
        acc = 0.0
        for j in range(n_state):
            acc += c[j] * x[j]
        y[i] = acc + d * u[i]

        ui = u[i]
        ui1 = u[i + 1]
        for r in range(n_state):
            val = b0[r] * ui + b1[r] * ui1
            for s in range(n_state):
                val += ad[r, s] * x[s]
            x_new[r] = val
        for r in range(n_state):
            x[r] = x_new[r]

    acc = 0.0
    for j in range(n_state):
        acc += c[j] * x[j]
    y[n_steps - 1] = acc + d * u[n_steps - 1]
    return y

class Solver:
    def __init__(self) -> None:
        self._system_cache: dict[
            tuple[bytes, bytes],
            tuple[np.ndarray, np.ndarray, np.ndarray, float, bool],
        ] = {}
        self._interp_cache: dict[
            tuple[bytes, bytes, float],
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
        ] = {}

        ad = np.eye(1, dtype=np.float64)
        z = np.zeros(1, dtype=np.float64)
        _simulate_ss_numba(ad, z, z, z, 0.0, z)

    def _system_key(self, num: np.ndarray, den: np.ndarray) -> tuple[bytes, bytes]:
        return (num.tobytes(), den.tobytes())

    def _interp_key(
        self,
        num: np.ndarray,
        den: np.ndarray,
        dt: float,
    ) -> tuple[bytes, bytes, float]:
        return (num.tobytes(), den.tobytes(), float(dt))

    def _get_system(
        self,
        num: np.ndarray,
        den: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
        key = self._system_key(num, den)
        cached = self._system_cache.get(key)
        if cached is not None:
            return cached

        if num.size <= den.size and den.size >= 1 and den[0] != 0.0:
            den0 = float(den[0])
            den_n = den / den0
            n = den_n.size - 1

            num_n = num / den0
            if num_n.size < n + 1:
                num_pad = np.zeros(n + 1, dtype=np.float64)
                num_pad[-num_n.size :] = num_n
            else:
                num_pad = np.asarray(num_n[-(n + 1) :], dtype=np.float64)

            d_scalar = float(num_pad[0])

            if n == 0:
                out = (
                    np.empty((0, 0), dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                    d_scalar,
                    abs(float(den_n[-1])) > 1e-14,
                )
                self._system_cache[key] = out
                return out

            a = np.zeros((n, n), dtype=np.float64)
            a[0, :] = -den_n[1:]
            if n > 1:
                a[1:, :-1] = np.eye(n - 1, dtype=np.float64)

            b = np.zeros(n, dtype=np.float64)
            b[0] = 1.0
            c = np.ascontiguousarray(num_pad[1:] - d_scalar * den_n[1:])
            out = (
                np.ascontiguousarray(a),
                np.ascontiguousarray(b),
                c,
                d_scalar,
                abs(float(den_n[-1])) > 1e-14,
            )
            self._system_cache[key] = out
            return out

        a, b, c, d = _tf2ss(num, den)
        d_scalar = float(np.asarray(d, dtype=np.float64).reshape(-1)[0])
        out = (
            np.ascontiguousarray(np.asarray(a, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(b, dtype=np.float64).reshape(-1)),
            np.ascontiguousarray(np.asarray(c, dtype=np.float64).reshape(-1)),
            d_scalar,
            abs(float(den[-1] / den[0])) > 1e-14 if den.size and den[0] != 0.0 else False,
        )
        self._system_cache[key] = out
        return out

    def _get_interp_ss(
        self,
        num: np.ndarray,
        den: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        key = self._interp_key(num, den, dt)
        cached = self._interp_cache.get(key)
        if cached is not None:
            return cached

        a, b, c, d_scalar, nonsingular = self._get_system(num, den)
        n = a.shape[0]

        if n == 0:
            out = (
                np.empty((0, 0), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                c,
                d_scalar,
            )
            self._interp_cache[key] = out
            return out

        if n == 1:
            a00 = float(a[0, 0])
            b00 = float(b[0])
            if abs(a00) > 1e-14:
                ad00 = float(np.exp(a00 * dt))
                phi1 = ((ad00 - 1.0) / a00) * b00
                phi2 = (dt * ad00 * b00 - phi1) / a00
                bd0 = phi2 / dt
                bd1 = phi1 - bd0
            else:
                ad00 = 1.0
                bd0 = 0.5 * dt * b00
                bd1 = bd0

            out = (
                np.array([[ad00]], dtype=np.float64),
                np.array([bd0], dtype=np.float64),
                np.array([bd1], dtype=np.float64),
                c,
                d_scalar,
            )
            self._interp_cache[key] = out
            return out

        if n == 2 and nonsingular:
            a11 = float(a[0, 0])
            a12 = float(a[0, 1])
            a21 = float(a[1, 0])
            a22 = float(a[1, 1])

            mu = 0.5 * (a11 + a22)
            m00 = a11 - mu
            m01 = a12
            m10 = a21
            m11 = a22 - mu

            delta = np.sqrt((m00 * m00 + m01 * m10) + 0j)
            z = delta * dt
            e = np.exp(mu * dt)

            if abs(delta) > 1e-14:
                ch = np.cosh(z)
                sh_over_delta = np.sinh(z) / delta
                ad_c = e * np.array(
                    [
                        [ch + sh_over_delta * m00, sh_over_delta * m01],
                        [sh_over_delta * m10, ch + sh_over_delta * m11],
                    ],
                    dtype=np.complex128,
                )
            else:
                ad_c = e * np.array(
                    [[1.0 + dt * m00, dt * m01], [dt * m10, 1.0 + dt * m11]],
                    dtype=np.complex128,
                )

            ad = np.ascontiguousarray(np.asarray(np.real_if_close(ad_c), dtype=np.float64))

            det = a11 * a22 - a12 * a21
            if abs(det) > 1e-14:
                v0 = ad[0, 0] * b[0] + ad[0, 1] * b[1]
                v1 = ad[1, 0] * b[0] + ad[1, 1] * b[1]
                r10 = v0 - b[0]
                r11 = v1 - b[1]

                inv_det = 1.0 / det
                phi10 = (a22 * r10 - a12 * r11) * inv_det
                phi11 = (-a21 * r10 + a11 * r11) * inv_det

                s0 = dt * v0 - phi10
                s1 = dt * v1 - phi11
                phi20 = (a22 * s0 - a12 * s1) * inv_det
                phi21 = (-a21 * s0 + a11 * s1) * inv_det

                bd0 = np.array([phi20 / dt, phi21 / dt], dtype=np.float64)
                bd1 = np.array([phi10 - bd0[0], phi11 - bd0[1]], dtype=np.float64)

                out = (ad, bd0, bd1, c, d_scalar)
                self._interp_cache[key] = out
                return out

        m = np.zeros((n + 2, n + 2), dtype=np.float64)
        m[:n, :n] = a * dt
        m[:n, n] = b * dt
        m[n, n + 1] = 1.0

        em = expm(m)
        ad = np.ascontiguousarray(np.asarray(em[:n, :n], dtype=np.float64))
        bd1 = np.ascontiguousarray(np.asarray(em[:n, n + 1], dtype=np.float64))
        bd0 = np.ascontiguousarray(np.asarray(em[:n, n] - em[:n, n + 1], dtype=np.float64))

        out = (ad, bd0, bd1, c, d_scalar)
        self._interp_cache[key] = out
        return out

    def solve(self, problem, **kwargs) -> Any:
        num = np.asarray(problem["num"], dtype=np.float64).ravel()
        den = np.asarray(problem["den"], dtype=np.float64).ravel()
        u = np.asarray(problem["u"], dtype=np.float64).ravel()
        t = np.asarray(problem["t"], dtype=np.float64).ravel()

        n = t.size
        if n == 0:
            return {"yout": []}

        if u.size != n:
            _, y, _ = _lsim((num, den), U=u, T=t)
            return {"yout": y.tolist()}

        if den.size == 1 and num.size == 1:
            return {"yout": (float(num[0] / den[0]) * u).tolist()}

        if n == 1:
            try:
                _, _, _, d_scalar, _ = self._get_system(num, den)
                return {"yout": [d_scalar * float(u[0])]}
            except Exception:
                _, y, _ = _lsim((num, den), U=u, T=t)
                return {"yout": y.tolist()}

        dt = float(t[1] - t[0])
        if n > 2:
            tol = 1e-12 + 1e-9 * abs(dt)
            if float(np.max(np.abs((t[2:] - t[1:-1]) - dt))) > tol:
                _, y, _ = _lsim((num, den), U=u, T=t)
                return {"yout": y.tolist()}

        try:
            ad, bd0, bd1, c, d_scalar = self._get_interp_ss(num, den, dt)
            y = _simulate_ss_numba(ad, bd0, bd1, c, d_scalar, u)
            return {"yout": y.tolist()}
        except Exception:
            _, y, _ = _lsim((num, den), U=u, T=t)
            return {"yout": y.tolist()}