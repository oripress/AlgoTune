from __future__ import annotations

from typing import Any

import numpy as np
import scipy.linalg as la
from numba import njit

@njit(cache=True)
def _eig2x2_nb(A, B):
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    e, f = B[0, 0], B[0, 1]
    g, h = B[1, 0], B[1, 1]

    vals = np.empty(2, dtype=np.complex128)
    vecs = np.empty((2, 2), dtype=np.complex128)

    c2 = e * h - f * g
    if abs(c2) < 1e-14:
        return False, vals, vecs

    c1 = -a * h - d * e + b * g + c * f
    c0 = a * d - b * c
    disc = np.complex128(c1 * c1 - 4.0 * c2 * c0)
    root = np.sqrt(disc)
    denom = 2.0 * c2
    vals[0] = (-c1 + root) / denom
    vals[1] = (-c1 - root) / denom

    for i in range(2):
        lam = vals[i]
        m11 = a - lam * e
        m12 = b - lam * f
        m21 = c - lam * g
        m22 = d - lam * h

        n1 = (m12.real * m12.real + m12.imag * m12.imag) + (m11.real * m11.real + m11.imag * m11.imag)
        n2 = (m22.real * m22.real + m22.imag * m22.imag) + (m21.real * m21.real + m21.imag * m21.imag)

        if n1 >= n2 and n1 > 1e-30:
            s = 1.0 / np.sqrt(n1)
            vecs[0, i] = -m12 * s
            vecs[1, i] = m11 * s
        elif n2 > 1e-30:
            s = 1.0 / np.sqrt(n2)
            vecs[0, i] = -m22 * s
            vecs[1, i] = m21 * s
        else:
            return False, vals, vecs

    if (vals[0].real < vals[1].real) or (vals[0].real == vals[1].real and vals[0].imag < vals[1].imag):
        tmp = vals[0]
        vals[0] = vals[1]
        vals[1] = tmp
        tmp0 = vecs[0, 0]
        tmp1 = vecs[1, 0]
        vecs[0, 0] = vecs[0, 1]
        vecs[1, 0] = vecs[1, 1]
        vecs[0, 1] = tmp0
        vecs[1, 1] = tmp1

    return True, vals, vecs

class Solver:
    def __init__(self):
        eye = np.eye(2, dtype=np.float64)
        _eig2x2_nb(eye, eye)

    def solve(self, problem, **kwargs) -> Any:
        A, B = problem
        if type(A) is not np.ndarray or A.dtype != np.float64:
            A = np.asarray(A, dtype=float)
        if type(B) is not np.ndarray or B.dtype != np.float64:
            B = np.asarray(B, dtype=float)

        n = A.shape[0]

        if n == 1:
            return [complex(A[0, 0] / B[0, 0])], [[1.0 + 0.0j]]

        if n == 2:
            ok, vals, vecs = _eig2x2_nb(A, B)
            if ok:
                return vals.tolist(), vecs.T.tolist()

        try:
            vals, vecs = np.linalg.eig(np.linalg.solve(B, A))
        except np.linalg.LinAlgError:
            vals, vecs = la.eig(
                A,
                B,
                left=False,
                right=True,
                check_finite=False,
            )
            norms = np.sqrt(np.sum(vecs.real * vecs.real + vecs.imag * vecs.imag, axis=0))
            norms = np.where(norms > 1e-15, norms, 1.0)
            vecs = vecs / norms

        order = np.lexsort((-vals.imag, -vals.real))
        return vals[order].tolist(), vecs[:, order].T.tolist()