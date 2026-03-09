from typing import Any

import numpy as np
from scipy.linalg.lapack import dgesv

class Solver:
    def __init__(self) -> None:
        self._array = np.array
        self._dgesv = dgesv
        self._dtype = np.float64

    @staticmethod
    def _small_solve(A, b, n: int) -> list[float]:
        m = []
        for i in range(n):
            row = list(A[i])
            row.append(b[i])
            m.append(row)

        for k in range(n):
            pivot_row = k
            pivot_abs = abs(m[k][k])
            for i in range(k + 1, n):
                v = abs(m[i][k])
                if v > pivot_abs:
                    pivot_abs = v
                    pivot_row = i
            if pivot_row != k:
                m[k], m[pivot_row] = m[pivot_row], m[k]

            rowk = m[k]
            pivot = rowk[k]
            for i in range(k + 1, n):
                rowi = m[i]
                factor = rowi[k] / pivot
                rowi[k] = 0.0
                for j in range(k + 1, n + 1):
                    rowi[j] -= factor * rowk[j]

        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            rowi = m[i]
            s = rowi[n]
            for j in range(i + 1, n):
                s -= rowi[j] * x[j]
            x[i] = s / rowi[i]
        return x

    def solve(self, problem, **kwargs) -> Any:
        A = problem["A"]
        b = problem["b"]
        n = len(b)
        array = self._array
        dtype = self._dtype
        dgesv = self._dgesv
        small_solve = self._small_solve

        if n == 1:
            return [b[0] / A[0][0]]

        if n == 2:
            r0 = A[0]
            r1 = A[1]
            a00 = r0[0]
            a01 = r0[1]
            a10 = r1[0]
            a11 = r1[1]
            b0 = b[0]
            b1 = b[1]
            det = a00 * a11 - a01 * a10
            return [
                (b0 * a11 - a01 * b1) / det,
                (a00 * b1 - b0 * a10) / det,
            ]

        if n == 3:
            r0 = A[0]
            r1 = A[1]
            r2 = A[2]
            a00 = r0[0]
            a01 = r0[1]
            a02 = r0[2]
            a10 = r1[0]
            a11 = r1[1]
            a12 = r1[2]
            a20 = r2[0]
            a21 = r2[1]
            a22 = r2[2]
            b0 = b[0]
            b1 = b[1]
            b2 = b[2]

            det = (
                a00 * (a11 * a22 - a12 * a21)
                - a01 * (a10 * a22 - a12 * a20)
                + a02 * (a10 * a21 - a11 * a20)
            )

            return [
                (
                    b0 * (a11 * a22 - a12 * a21)
                    - a01 * (b1 * a22 - a12 * b2)
                    + a02 * (b1 * a21 - a11 * b2)
                ) / det,
                (
                    a00 * (b1 * a22 - a12 * b2)
                    - b0 * (a10 * a22 - a12 * a20)
                    + a02 * (a10 * b2 - b1 * a20)
                ) / det,
                (
                    a00 * (a11 * b2 - b1 * a21)
                    - a01 * (a10 * b2 - b1 * a20)
                    + b0 * (a10 * a21 - a11 * a20)
                ) / det,
            ]

        if n <= 8:
            return small_solve(A, b, n)

        a_arr = array(A, dtype=dtype, order="F")
        b_arr = array(b, dtype=dtype).reshape(n, 1)
        _, _, x, info = dgesv(
            a_arr,
            b_arr,
            overwrite_a=True,
            overwrite_b=True,
        )
        if info != 0:
            raise np.linalg.LinAlgError("Singular matrix")
        return x[:, 0]