from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit
from scipy.linalg import expm

@njit(cache=True)
def _simulate_constdt_foh_n1(
    Ad00: float,
    Bd0: float,
    Bd1: float,
    C0: float,
    D: float,
    u: np.ndarray,
) -> np.ndarray:
    N = u.shape[0]
    y = np.empty(N, dtype=np.float64)
    x = 0.0
    y[0] = D * u[0]
    for k in range(N - 1):
        uk = u[k]
        uk1 = u[k + 1]
        x = Ad00 * x + Bd0 * uk + Bd1 * uk1
        y[k + 1] = C0 * x + D * uk1
    return y

@njit(cache=True)
def _simulate_constdt_foh_n2(
    Ad00: float,
    Ad01: float,
    Ad10: float,
    Ad11: float,
    Bd00: float,
    Bd01: float,
    Bd10: float,
    Bd11: float,
    C0: float,
    C1: float,
    D: float,
    u: np.ndarray,
) -> np.ndarray:
    N = u.shape[0]
    y = np.empty(N, dtype=np.float64)
    x0 = 0.0
    x1 = 0.0
    y[0] = D * u[0]
    for k in range(N - 1):
        uk = u[k]
        uk1 = u[k + 1]
        nx0 = Ad00 * x0 + Ad01 * x1 + Bd00 * uk + Bd01 * uk1
        nx1 = Ad10 * x0 + Ad11 * x1 + Bd10 * uk + Bd11 * uk1
        x0 = nx0
        x1 = nx1
        y[k + 1] = C0 * x0 + C1 * x1 + D * uk1
    return y

@njit(cache=True)
def _simulate_constdt_foh(
    Ad: np.ndarray,
    Bd0: np.ndarray,
    Bd1: np.ndarray,
    C: np.ndarray,
    D: float,
    u: np.ndarray,
) -> np.ndarray:
    n = Ad.shape[0]
    N = u.shape[0]
    y = np.empty(N, dtype=np.float64)

    x = np.zeros(n, dtype=np.float64)
    x_new = np.empty(n, dtype=np.float64)
    y[0] = D * u[0]

    for k in range(N - 1):
        uk = u[k]
        uk1 = u[k + 1]

        for i in range(n):
            s = 0.0
            for j in range(n):
                s += Ad[i, j] * x[j]
            s += Bd0[i] * uk + Bd1[i] * uk1
            x_new[i] = s

        tmp = x
        x = x_new
        x_new = tmp

        s = 0.0
        for j in range(n):
            s += C[j] * x[j]
        y[k + 1] = s + D * uk1

    return y

def _expm_2x2(Z: np.ndarray) -> np.ndarray:
    a = float(Z[0, 0])
    b = float(Z[0, 1])
    c = float(Z[1, 0])
    d = float(Z[1, 1])

    tr = 0.5 * (a + d)
    det = a * d - b * c
    disc = tr * tr - det

    etr = math.exp(tr)
    n00 = a - tr
    n11 = d - tr

    if disc > 0.0:
        s = math.sqrt(disc)
        if s > 1e-12:
            cosh_s = math.cosh(s)
            sinhc = math.sinh(s) / s
        else:
            cosh_s = 1.0 + 0.5 * disc
            sinhc = 1.0 + disc / 6.0
        e00 = etr * (cosh_s + sinhc * n00)
        e11 = etr * (cosh_s + sinhc * n11)
        e01 = etr * (sinhc * b)
        e10 = etr * (sinhc * c)
    elif disc < 0.0:
        w2 = -disc
        w = math.sqrt(w2)
        if w > 1e-12:
            cos_w = math.cos(w)
            sinc = math.sin(w) / w
        else:
            cos_w = 1.0 - 0.5 * w2
            sinc = 1.0 - w2 / 6.0
        e00 = etr * (cos_w + sinc * n00)
        e11 = etr * (cos_w + sinc * n11)
        e01 = etr * (sinc * b)
        e10 = etr * (sinc * c)
    else:
        e00 = etr * (1.0 + n00)
        e11 = etr * (1.0 + n11)
        e01 = etr * b
        e10 = etr * c

    return np.array([[e00, e01], [e10, e11]], dtype=np.float64)

def _solve_companion(A_last: np.ndarray, f: np.ndarray) -> np.ndarray:
    n = f.size
    x = np.empty(n, dtype=np.float64)
    if n > 1:
        x[1:] = f[:-1]
        rest = float(np.dot(A_last[1:], x[1:]))
    else:
        rest = 0.0
    a_n = -float(A_last[0])
    x[0] = (rest - float(f[-1])) / a_n
    return x

def _trim_leading_zeros(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    i = 0
    n = x.size
    while i < n and x[i] == 0:
        i += 1
    if i == 0:
        return x
    if i == n:
        return np.array([0.0], dtype=np.float64)
    return x[i:]

def _tf2ss_siso(num: np.ndarray, den: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    den = _trim_leading_zeros(den)
    num = _trim_leading_zeros(num)

    if den.size == 0 or den[0] == 0:
        raise ValueError("Invalid denominator")

    den0 = den[0]
    den = den / den0
    num = num / den0

    if num.size > den.size:
        raise ValueError("Improper transfer function")

    n = den.size - 1
    if n == 0:
        return (
            np.zeros((0, 0), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            float(num[0]),
            True,
        )

    A_invertible = den[-1] != 0.0

    if num.size < n + 1:
        num_padded = np.zeros(n + 1, dtype=np.float64)
        num_padded[(n + 1 - num.size) :] = num
    else:
        num_padded = num

    A = np.zeros((n, n), dtype=np.float64)
    if n > 1:
        A[:-1, 1:] = np.eye(n - 1, dtype=np.float64)
    A[-1, :] = -den[1:][::-1]

    B = np.zeros(n, dtype=np.float64)
    B[-1] = 1.0

    C = (num_padded[1:] - num_padded[0] * den[1:])[::-1].astype(np.float64, copy=False)
    D = float(num_padded[0])
    return A, B, C, D, A_invertible

class Solver:
    def __init__(self) -> None:
        # compile numba kernels (excluded from solve runtime)
        u = np.ones(2, dtype=np.float64)
        _simulate_constdt_foh(np.eye(1, dtype=np.float64), np.ones(1), np.ones(1), np.ones(1), 0.0, u)
        _simulate_constdt_foh_n1(1.0, 1.0, 1.0, 1.0, 0.0, u)
        _simulate_constdt_foh_n2(1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, u)

    def solve(self, problem, **kwargs) -> Any:
        num = np.asarray(problem["num"], dtype=np.float64)
        den = np.asarray(problem["den"], dtype=np.float64)
        u = np.asarray(problem["u"], dtype=np.float64)
        t = np.asarray(problem["t"], dtype=np.float64)

        N = t.size
        if N == 0:
            return {"yout": []}
        if u.size != N:
            u = u[:N]

        try:
            A, B, C, D, A_invertible = _tf2ss_siso(num, den)
        except ValueError:
            from scipy import signal

            system = signal.lti(num, den)
            _, yout, _ = signal.lsim(system, u, t)
            return {"yout": yout.tolist()}

        if A.size == 0:
            return {"yout": (D * u).astype(np.float64, copy=False).tolist()}

        if N == 1:
            return {"yout": [float(D * u[0])]}

        dt = np.diff(t)
        dt0 = float(dt[0])

        if np.allclose(dt, dt0, rtol=1e-12, atol=1e-15):
            n = A.shape[0]

            if A_invertible:
                if n == 1:
                    z = float(A[0, 0]) * dt0
                    e = float(math.exp(z))
                    if abs(z) > 1e-12:
                        phi1 = (e - 1.0) / z
                        phi2 = (e - 1.0 - z) / (z * z)
                    else:
                        phi1 = 1.0 + 0.5 * z
                        phi2 = 0.5 + z / 6.0

                    Bd1 = dt0 * phi2
                    Bd0 = dt0 * (phi1 - phi2)
                    y = _simulate_constdt_foh_n1(e, Bd0, Bd1, float(C[0]), D, u)
                    return {"yout": y.tolist()}

                Z = A * dt0
                if n == 2:
                    Ad = _expm_2x2(Z)

                    a2 = -float(A[1, 0])
                    a1 = -float(A[1, 1])

                    Ad00 = float(Ad[0, 0])
                    Ad01 = float(Ad[0, 1])
                    Ad10 = float(Ad[1, 0])
                    Ad11 = float(Ad[1, 1])

                    F1_0 = Ad01
                    F1_1 = Ad11 - 1.0

                    F2_0 = Ad01 - dt0
                    F2_1 = Ad11 - 1.0 + a1 * dt0

                    v1_1 = F1_0
                    v1_0 = (-a1 * F1_0 - F1_1) / a2

                    w1 = F2_0
                    w0 = (-a1 * F2_0 - F2_1) / a2

                    v2_1 = w0
                    v2_0 = (-a1 * w0 - w1) / a2

                    Bd1_0 = v2_0 / dt0
                    Bd1_1 = v2_1 / dt0
                    Bd0_0 = v1_0 - Bd1_0
                    Bd0_1 = v1_1 - Bd1_1

                    y = _simulate_constdt_foh_n2(
                        Ad00,
                        Ad01,
                        Ad10,
                        Ad11,
                        Bd0_0,
                        Bd1_0,
                        Bd0_1,
                        Bd1_1,
                        float(C[0]),
                        float(C[1]),
                        D,
                        u,
                    )
                    return {"yout": y.tolist()}

                Ad = expm(Z)
                AdB = Ad[:, -1]
                AB = A[:, -1]
                F1 = AdB - B
                F2 = AdB - B - AB * dt0

                A_last = A[-1]
                v1 = _solve_companion(A_last, F1)
                w = _solve_companion(A_last, F2)
                v2 = _solve_companion(A_last, w)

                Bd1v = v2 / dt0
                Bd0v = v1 - Bd1v
                y = _simulate_constdt_foh(Ad, Bd0v, Bd1v, C, D, u)
                return {"yout": y.tolist()}

            # singular A: augmented expm
            M = np.zeros((n + 2, n + 2), dtype=np.float64)
            M[:n, :n] = A
            M[:n, n] = B
            M[n, n + 1] = 1.0
            expM = expm(M * dt0)

            Ad = expM[:n, :n]
            Phi12 = expM[:n, n]
            Phi13 = expM[:n, n + 1]
            Bd1 = Phi13 / dt0
            Bd0 = Phi12 - Bd1

            y = _simulate_constdt_foh(Ad, Bd0, Bd1, C, D, u)
            return {"yout": y.tolist()}

        # variable dt: augmented expm per unique dt (rare)
        n = A.shape[0]
        Mbase = np.zeros((n + 2, n + 2), dtype=np.float64)
        Mbase[:n, :n] = A
        Mbase[:n, n] = B
        Mbase[n, n + 1] = 1.0

        x = np.zeros(n, dtype=np.float64)
        y = np.empty(N, dtype=np.float64)
        y[0] = D * u[0]

        step_cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for k in range(N - 1):
            dtk = float(dt[k])
            uk = float(u[k])
            uk1 = float(u[k + 1])

            mats = step_cache.get(dtk)
            if mats is None:
                expM = expm(Mbase * dtk)
                Adk = expM[:n, :n]
                Phi12 = expM[:n, n]
                Phi13 = expM[:n, n + 1]
                Bd1k = Phi13 / dtk
                Bd0k = Phi12 - Bd1k
                mats = (Adk, Bd0k, Bd1k)
                step_cache[dtk] = mats

            Adk, Bd0k, Bd1k = mats
            x = Adk.dot(x) + Bd0k * uk + Bd1k * uk1
            y[k + 1] = C.dot(x) + D * uk1

        return {"yout": y.tolist()}