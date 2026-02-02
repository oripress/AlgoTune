from __future__ import annotations

from typing import Any

import numpy as np

try:  # optional fast path
    from scipy.linalg.lapack import dgesv as _dgesv  # type: ignore
except Exception:  # pragma: no cover
    _dgesv = None

class Solver:
    __slots__ = ("_solve", "_array", "_asfortranarray", "_dgesv")

    def __init__(self) -> None:
        # Cache hot callables to reduce per-call attribute/global lookup overhead.
        self._solve = np.linalg.solve
        self._array = np.array
        self._asfortranarray = np.asfortranarray
        self._dgesv = _dgesv

    @staticmethod
    def _solve_1x1_list(A: list[list[Any]], b: list[Any]) -> list[float]:
        return [float(b[0]) / float(A[0][0])]

    @staticmethod
    def _solve_2x2_list(A: list[list[Any]], b: list[Any]) -> list[float]:
        r0 = A[0]
        r1 = A[1]
        a00 = float(r0[0])
        a01 = float(r0[1])
        a10 = float(r1[0])
        a11 = float(r1[1])
        b0 = float(b[0])
        b1 = float(b[1])
        det = a00 * a11 - a01 * a10
        return [(b0 * a11 - b1 * a01) / det, (a00 * b1 - a10 * b0) / det]

    @staticmethod
    def _solve_3x3_list(A: list[list[Any]], b: list[Any]) -> list[float]:
        r0 = A[0]
        r1 = A[1]
        r2 = A[2]
        a00 = float(r0[0])
        a01 = float(r0[1])
        a02 = float(r0[2])
        a10 = float(r1[0])
        a11 = float(r1[1])
        a12 = float(r1[2])
        a20 = float(r2[0])
        a21 = float(r2[1])
        a22 = float(r2[2])
        b0 = float(b[0])
        b1 = float(b[1])
        b2 = float(b[2])

        c00 = a11 * a22 - a12 * a21
        c01 = -(a10 * a22 - a12 * a20)
        c02 = a10 * a21 - a11 * a20
        det = a00 * c00 + a01 * c01 + a02 * c02

        c10 = -(a01 * a22 - a02 * a21)
        c11 = a00 * a22 - a02 * a20
        c12 = -(a00 * a21 - a01 * a20)

        c20 = a01 * a12 - a02 * a11
        c21 = -(a00 * a12 - a02 * a10)
        c22 = a00 * a11 - a01 * a10

        x0 = (c00 * b0 + c10 * b1 + c20 * b2) / det
        x1 = (c01 * b0 + c11 * b1 + c21 * b2) / det
        x2 = (c02 * b0 + c12 * b1 + c22 * b2) / det
        return [x0, x1, x2]

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        A = problem["A"]
        b = problem["b"]

        # Localize hot attributes (micro-optimization for Python overhead).
        solve = self._solve
        array = self._array
        asfortranarray = self._asfortranarray
        dgesv = self._dgesv

        # Tiny fast path for typical JSON/list inputs.
        if not isinstance(b, np.ndarray) and not isinstance(A, np.ndarray):
            n = len(b)
            if n == 1:
                return self._solve_1x1_list(A, b)  # type: ignore[arg-type]
            if n == 2:
                return self._solve_2x2_list(A, b)  # type: ignore[arg-type]
            if n == 3:
                return self._solve_3x3_list(A, b)  # type: ignore[arg-type]

            # For larger n, build A directly in Fortran order.
            a_arr = array(A, dtype=np.float64, order="F")
            b_arr = array(b, dtype=np.float64)

            # Direct LAPACK call can be slightly lower overhead than np.linalg.solve.
            if dgesv is not None:
                b2 = b_arr.reshape((n, 1))
                _, _, x2, info = dgesv(a_arr, b2, overwrite_a=True, overwrite_b=True)
                if info == 0:
                    return x2[:, 0].tolist()

            return solve(a_arr, b_arr).tolist()

        # ndarray / mixed inputs: coerce cheaply and ensure Fortran contiguity.
        if isinstance(A, np.ndarray):
            a_arr = A
            if a_arr.dtype != np.float64:
                a_arr = a_arr.astype(np.float64, copy=False)
            if not a_arr.flags["F_CONTIGUOUS"]:
                a_arr = asfortranarray(a_arr)
        else:
            a_arr = array(A, dtype=np.float64, order="F")

        if isinstance(b, np.ndarray):
            b_arr = b
            if b_arr.dtype != np.float64:
                b_arr = b_arr.astype(np.float64, copy=False)
        else:
            b_arr = array(b, dtype=np.float64)

        x = solve(a_arr, b_arr)
        return x.tolist()