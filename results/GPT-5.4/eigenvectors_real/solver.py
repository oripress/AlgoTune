from __future__ import annotations

import math
from typing import Any

import numba
import numpy as np

@numba.njit(cache=True)
def _jacobi_eigh_numba(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    D = A.copy()
    V = np.eye(n, dtype=np.float64)

    for _ in range(8 * n * n):
        max_val = 0.0
        p = 0
        q = 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                val = abs(D[i, j])
                if val > max_val:
                    max_val = val
                    p = i
                    q = j

        if max_val < 1e-12:
            break

        app = D[p, p]
        aqq = D[q, q]
        apq = D[p, q]

        tau = (aqq - app) / (2.0 * apq)
        if tau >= 0.0:
            t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
        else:
            t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        for k in range(n):
            if k != p and k != q:
                dkp = D[k, p]
                dkq = D[k, q]
                D[k, p] = c * dkp - s * dkq
                D[p, k] = D[k, p]
                D[k, q] = s * dkp + c * dkq
                D[q, k] = D[k, q]

        D[p, p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        D[q, q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        D[p, q] = 0.0
        D[q, p] = 0.0

        for k in range(n):
            vkp = V[k, p]
            vkq = V[k, q]
            V[k, p] = c * vkp - s * vkq
            V[k, q] = s * vkp + c * vkq

    w = np.empty(n, dtype=np.float64)
    for i in range(n):
        w[i] = D[i, i]
    return w, V

def _cross(
    u0: float, u1: float, u2: float, v0: float, v1: float, v2: float
) -> tuple[float, float, float]:
    return (u1 * v2 - u2 * v1, u2 * v0 - u0 * v2, u0 * v1 - u1 * v0)

def _eigenvector_3x3(A: np.ndarray, lam: float) -> tuple[float, float, float] | None:
    m00 = float(A[0, 0] - lam)
    m01 = float(A[0, 1])
    m02 = float(A[0, 2])
    m10 = float(A[1, 0])
    m11 = float(A[1, 1] - lam)
    m12 = float(A[1, 2])
    m20 = float(A[2, 0])
    m21 = float(A[2, 1])
    m22 = float(A[2, 2] - lam)

    c1 = _cross(m00, m01, m02, m10, m11, m12)
    c2 = _cross(m00, m01, m02, m20, m21, m22)
    c3 = _cross(m10, m11, m12, m20, m21, m22)

    n1 = c1[0] * c1[0] + c1[1] * c1[1] + c1[2] * c1[2]
    n2 = c2[0] * c2[0] + c2[1] * c2[1] + c2[2] * c2[2]
    n3 = c3[0] * c3[0] + c3[1] * c3[1] + c3[2] * c3[2]

    if n1 >= n2 and n1 >= n3:
        x, y, z = c1
        n = n1
    elif n2 >= n3:
        x, y, z = c2
        n = n2
    else:
        x, y, z = c3
        n = n3

    if n <= 1e-30:
        return None

    inv = 1.0 / math.sqrt(n)
    return (x * inv, y * inv, z * inv)

def _solve_3x3(A: np.ndarray) -> tuple[list[float], list[list[float]]] | None:
    a = float(A[0, 0])
    b = float(A[0, 1])
    c = float(A[0, 2])
    d = float(A[1, 1])
    e = float(A[1, 2])
    f = float(A[2, 2])

    p1 = b * b + c * c + e * e
    if p1 == 0.0:
        vals = [a, d, f]
        idx = sorted(range(3), key=vals.__getitem__, reverse=True)
        basis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        return ([vals[i] for i in idx], [basis[i] for i in idx])

    tr = a + d + f
    q = tr / 3.0
    aq = a - q
    dq = d - q
    fq = f - q
    p2 = aq * aq + dq * dq + fq * fq + 2.0 * p1
    p = math.sqrt(p2 / 6.0)
    if p == 0.0:
        return ([q, q, q], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    bp00 = aq / p
    bp01 = b / p
    bp02 = c / p
    bp11 = dq / p
    bp12 = e / p
    bp22 = fq / p

    detb = (
        bp00 * (bp11 * bp22 - bp12 * bp12)
        - bp01 * (bp01 * bp22 - bp12 * bp02)
        + bp02 * (bp01 * bp12 - bp11 * bp02)
    )
    r = 0.5 * detb
    if r <= -1.0:
        phi = math.pi / 3.0
    elif r >= 1.0:
        phi = 0.0
    else:
        phi = math.acos(r) / 3.0

    two_p = 2.0 * p
    lam1 = q + two_p * math.cos(phi)
    lam3 = q + two_p * math.cos(phi + 2.0 * math.pi / 3.0)
    lam2 = tr - lam1 - lam3

    v1 = _eigenvector_3x3(A, lam1)
    v3 = _eigenvector_3x3(A, lam3)
    if v1 is None or v3 is None:
        return None

    dot13 = v1[0] * v3[0] + v1[1] * v3[1] + v1[2] * v3[2]
    x3 = v3[0] - dot13 * v1[0]
    y3 = v3[1] - dot13 * v1[1]
    z3 = v3[2] - dot13 * v1[2]
    n3 = x3 * x3 + y3 * y3 + z3 * z3
    if n3 <= 1e-24:
        return None
    inv3 = 1.0 / math.sqrt(n3)
    v3 = (x3 * inv3, y3 * inv3, z3 * inv3)

    v2 = _cross(v3[0], v3[1], v3[2], v1[0], v1[1], v1[2])
    n2 = v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]
    if n2 <= 1e-24:
        return None
    inv2 = 1.0 / math.sqrt(n2)
    v2 = (v2[0] * inv2, v2[1] * inv2, v2[2] * inv2)

    vals = [lam1, lam2, lam3]
    vecs = [v1, v2, v3]

    norm_a = math.sqrt(a * a + d * d + f * f + 2.0 * (b * b + c * c + e * e)) + 1e-12
    for lam, v in zip(vals, vecs):
        r0 = a * v[0] + b * v[1] + c * v[2] - lam * v[0]
        r1 = b * v[0] + d * v[1] + e * v[2] - lam * v[1]
        r2 = c * v[0] + e * v[1] + f * v[2] - lam * v[2]
        if math.sqrt(r0 * r0 + r1 * r1 + r2 * r2) / norm_a > 1e-8:
            return None

    return (
        [lam1, lam2, lam3],
        [
            [v1[0], v1[1], v1[2]],
            [v2[0], v2[1], v2[2]],
            [v3[0], v3[1], v3[2]],
        ],
    )

class Solver:
    def __init__(self) -> None:
        _jacobi_eigh_numba(np.eye(4, dtype=np.float64))

    def solve(self, problem, **kwargs) -> Any:
        A = problem if isinstance(problem, np.ndarray) else np.asarray(problem)
        n = A.shape[0]

        if n == 0:
            return ([], [])

        if n == 1:
            return ([float(A[0, 0])], [[1.0]])

        if n == 2:
            a = float(A[0, 0])
            b = float(A[0, 1])
            d = float(A[1, 1])

            if b == 0.0:
                if a >= d:
                    return ([a, d], [[1.0, 0.0], [0.0, 1.0]])
                return ([d, a], [[0.0, 1.0], [1.0, 0.0]])

            tr = a + d
            diff = a - d
            disc = math.hypot(diff, 2.0 * b)
            lam1 = 0.5 * (tr + disc)
            lam2 = 0.5 * (tr - disc)

            x = b
            y = lam1 - a
            norm = math.hypot(x, y)
            if norm == 0.0:
                x = lam1 - d
                y = b
                norm = math.hypot(x, y)
            inv = 1.0 / norm
            x *= inv
            y *= inv

            return ([lam1, lam2], [[x, y], [-y, x]])

        if n == 3:
            out = _solve_3x3(A)
            if out is not None:
                return out

        if n <= 6:
            Af = np.asarray(A, dtype=np.float64)
            vals, vecs = _jacobi_eigh_numba(Af)
            idx = np.argsort(vals)[::-1]
            return (vals[idx].tolist(), vecs[:, idx].T.tolist())

        vals, vecs = np.linalg.eigh(A)
        return (vals[::-1].tolist(), vecs[:, ::-1].T.tolist())