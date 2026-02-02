from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

class Solver:
    __slots__ = (
        "_asarray",
        "_dot",
        "_empty",
        "_result_type",
        "_buf",
        "_buf_shape",
        "_buf_dtype",
    )

    def __init__(self) -> None:
        # Bind lookups once.
        self._asarray = np.asarray
        self._dot = np.dot
        self._empty = np.empty
        self._result_type = np.result_type

        # Optional reusable output buffer (helps when shapes repeat).
        self._buf: Optional[np.ndarray] = None
        self._buf_shape: Optional[Tuple[int, int]] = None
        self._buf_dtype: Any = None

    @staticmethod
    def _py_matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
        # B as columns for cache-friendly access.
        B_cols = list(zip(*B))
        out: list[list[float]] = []
        for ar in A:
            row_out = []
            for bc in B_cols:
                s = 0.0
                for a, b in zip(ar, bc):
                    s += a * b
                row_out.append(s)
            out.append(row_out)
        return out

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        A0 = problem["A"]
        B0 = problem["B"]

        # Fast path for tiny matrices given as Python lists: avoid NumPy conversion overhead.
        if type(A0) is list and type(B0) is list and A0 and B0:
            n = len(A0)
            k = len(A0[0])
            p = len(B0[0])
            # Heuristic threshold tuned for typical overhead crossover.
            if n * k * p <= 2048:
                return self._py_matmul(A0, B0)

        # NumPy/BLAS path.
        A = self._asarray(A0)
        B = self._asarray(B0)

        n = A.shape[0]
        p = B.shape[1]
        dtype = self._result_type(A, B)

        buf = self._buf
        if buf is None or self._buf_shape != (n, p) or self._buf_dtype != dtype:
            buf = self._empty((n, p), dtype=dtype)
            self._buf = buf
            self._buf_shape = (n, p)
            self._buf_dtype = dtype

        self._dot(A, B, out=buf)
        return buf