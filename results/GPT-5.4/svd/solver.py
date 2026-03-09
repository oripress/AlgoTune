from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg.lapack import dgesdd, sgesdd

class Solver:
    def __init__(self) -> None:
        self._dgesdd = dgesdd
        self._sgesdd = sgesdd
        self._np_svd = np.linalg.svd
        self._eigh = np.linalg.eigh
        self._asarray = np.asarray
        self._norm = np.linalg.norm
        self._empty = np.empty
        self._array = np.array
        self._zeros = np.zeros
        self._float = np.float64
        self._eps = np.finfo(np.float64).eps

    def _lapack_svd_ndarray(self, A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if A.dtype == np.float32:
            if not (A.flags.f_contiguous and A.flags.writeable):
                A = np.array(A, dtype=np.float32, order="F", copy=True)
            U, s, Vt, info = self._sgesdd(A, compute_uv=1, full_matrices=0, overwrite_a=1)
        else:
            if A.dtype != np.float64 or not (A.flags.f_contiguous and A.flags.writeable):
                A = np.array(A, dtype=np.float64, order="F", copy=True)
            U, s, Vt, info = self._dgesdd(A, compute_uv=1, full_matrices=0, overwrite_a=1)
        if info != 0:
            U, s, Vt = self._np_svd(A, full_matrices=False)
            return U, s, Vt.T
        return U, s, Vt.T

    def _complete_columns(self, Q: np.ndarray, start: int) -> None:
        n, k = Q.shape
        for i in range(n):
            if start >= k:
                return
            v = np.zeros(n, dtype=Q.dtype)
            v[i] = 1.0
            if start:
                v -= Q[:, :start] @ (Q[:, :start].T @ v)
            nv = float(self._norm(v))
            if nv > 1e-12:
                Q[:, start] = v / nv
                start += 1
        while start < k:
            v = np.zeros(n, dtype=Q.dtype)
            v[start % n] = 1.0
            if start:
                v -= Q[:, :start] @ (Q[:, :start].T @ v)
            nv = float(self._norm(v))
            if nv <= 1e-12:
                v = np.random.default_rng(start).standard_normal(n)
                v -= Q[:, :start] @ (Q[:, :start].T @ v)
                nv = float(self._norm(v))
            Q[:, start] = v / nv
            start += 1

    def _gram_svd(self, A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, m = A.shape
        if n >= m:
            G = A.T @ A
            w, V = self._eigh(G)
            w = np.maximum(w[::-1], 0.0)
            V = V[:, ::-1]
            s = np.sqrt(w)
            tol = self._eps * max(n, m) * max(float(s[0]), 1.0) * 8.0
            pos = int(np.sum(s > tol))
            U = np.empty((n, m), dtype=A.dtype)
            if pos:
                U[:, :pos] = (A @ V[:, :pos]) / s[:pos]
            if pos < m:
                self._complete_columns(U, pos)
            return U, s, V

        G = A @ A.T
        w, U = self._eigh(G)
        w = np.maximum(w[::-1], 0.0)
        U = U[:, ::-1]
        s = np.sqrt(w)
        tol = self._eps * max(n, m) * max(float(s[0]), 1.0) * 8.0
        pos = int(np.sum(s > tol))
        V = np.empty((m, n), dtype=A.dtype)
        if pos:
            V[:, :pos] = (A.T @ U[:, :pos]) / s[:pos]
        if pos < n:
            self._complete_columns(V, pos)
        return U, s, V

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        matrix = problem["matrix"]
        try:
            n = problem["n"]
            m = problem["m"]
        except KeyError:
            try:
                n, m = matrix.shape
            except AttributeError:
                n, m = self._asarray(matrix).shape

        if n == 0 or m == 0:
            return {
                "U": self._empty((n, 0), dtype=self._float),
                "S": self._empty((0,), dtype=self._float),
                "V": self._empty((m, 0), dtype=self._float),
            }

        if n == 1:
            row = self._asarray(matrix[0], dtype=self._float)
            s0 = float(self._norm(row))
            U = self._array([[1.0]], dtype=self._float)
            S = self._array([s0], dtype=self._float)
            V = self._zeros((m, 1), dtype=self._float)
            if s0 > 0.0:
                V[:, 0] = row / s0
            else:
                V[0, 0] = 1.0
            return {"U": U, "S": S, "V": V}

        if m == 1:
            col = self._empty(n, dtype=self._float)
            for i in range(n):
                col[i] = matrix[i][0]
            s0 = float(self._norm(col))
            U = self._zeros((n, 1), dtype=self._float)
            if s0 > 0.0:
                U[:, 0] = col / s0
            else:
                U[0, 0] = 1.0
            return {
                "U": U,
                "S": self._array([s0], dtype=self._float),
                "V": self._array([[1.0]], dtype=self._float),
            }

        if (n >= 2 * m or m >= 2 * n) and min(n, m) <= 64:
            A = self._asarray(matrix, dtype=self._float)
            U, s, V = self._gram_svd(A)
            return {"U": U, "S": s, "V": V}

        if isinstance(matrix, np.ndarray):
            U, s, V = self._lapack_svd_ndarray(matrix)
        else:
            U, s, Vt = self._np_svd(matrix, full_matrices=False)
            V = Vt.T
        return {"U": U, "S": s, "V": V}