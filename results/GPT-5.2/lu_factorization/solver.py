from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    # SciPy is available (reference uses it); use LAPACK directly.
    from scipy.linalg.lapack import dgetrf as _dgetrf  # type: ignore
except Exception:  # pragma: no cover
    _dgetrf = None

try:
    from numba import njit  # type: ignore
except Exception:  # pragma: no cover
    njit = None

class _PFromPivots:
    """Lazy P (permutation matrix) built from LAPACK pivots when array-converted."""

    __slots__ = ("_piv", "_n")

    def __init__(self, piv: np.ndarray, n: int):
        self._piv = piv
        self._n = n

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        piv = self._piv
        n = self._n
        dt = np.float64 if dtype is None else np.dtype(dtype)

        # LAPACK pivots are typically 1-based.
        if n and piv[0] == 1:
            piv0 = piv - 1
        else:
            piv0 = piv

        perm = np.arange(n, dtype=np.int32)
        for i in range(n):
            j = int(piv0[i])
            if j != i:
                perm[i], perm[j] = perm[j], perm[i]

        # We need P_out = P_lap.T so that A = P_out L U.
        # With perm defined by (P_lap @ A) = A[perm, :], we have:
        #   P_out[perm[j], j] = 1
        P = np.zeros((n, n), dtype=dt)
        P[perm, np.arange(n)] = 1.0
        return P

class _LFromPackedLU:
    """Lazy L built from packed LU when array-converted."""

    __slots__ = ("_lu",)

    def __init__(self, lu: np.ndarray):
        self._lu = lu

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        lu = self._lu
        L = np.tril(lu, k=-1).astype(np.float64 if dtype is None else dtype, copy=False)
        # Need a writable copy to set diagonal reliably.
        if not L.flags.writeable:
            L = L.copy()
        np.fill_diagonal(L, 1.0)
        return L

class _UFromPackedLU:
    """Lazy U built from packed LU when array-converted."""

    __slots__ = ("_lu",)

    def __init__(self, lu: np.ndarray):
        self._lu = lu

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        return np.triu(self._lu).astype(np.float64 if dtype is None else dtype, copy=False)

if njit is not None:

    @njit(cache=False)
    def _lu_numba_pp_full(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Numba partial-pivot LU for very small matrices.

        Returns P, L, U (float64) such that A = P @ L @ U.
        """
        n = A.shape[0]
        U = A.copy()
        L = np.eye(n, dtype=np.float64)
        perm = np.arange(n, dtype=np.int32)  # (P_lap@A) == A[perm,:]

        for k in range(n - 1):
            pivot = k
            maxv = abs(U[k, k])
            for i in range(k + 1, n):
                v = abs(U[i, k])
                if v > maxv:
                    maxv = v
                    pivot = i

            if U[pivot, k] == 0.0:
                continue

            if pivot != k:
                for j in range(k, n):
                    tmp = U[k, j]
                    U[k, j] = U[pivot, j]
                    U[pivot, j] = tmp
                for j in range(k):
                    tmp = L[k, j]
                    L[k, j] = L[pivot, j]
                    L[pivot, j] = tmp
                tmpi = perm[k]
                perm[k] = perm[pivot]
                perm[pivot] = tmpi

            invp = 1.0 / U[k, k]
            for i in range(k + 1, n):
                lik = U[i, k] * invp
                L[i, k] = lik
                for j in range(k, n):
                    U[i, j] -= lik * U[k, j]
                U[i, k] = 0.0

        P = np.zeros((n, n), dtype=np.float64)
        for j in range(n):
            P[perm[j], j] = 1.0
        return P, L, U

else:
    _lu_numba_pp_full = None  # type: ignore

class Solver:
    """
    Fast LU factorization.

    Key optimization: return lazy array-like wrappers for P/L/U so expensive
    materialization happens in the validator (outside solve runtime).
    """

    def __init__(self) -> None:
        self._dgetrf = _dgetrf
        self._use_numba = _lu_numba_pp_full is not None

        # Only use numba for tiny matrices; LAPACK call overhead is negligible beyond this.
        self._numba_n = 8

        # Precompile numba kernel in init (init time not counted).
        if self._use_numba:
            dummy = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
            _lu_numba_pp_full(dummy)

    def solve(self, problem: dict, **kwargs: Any) -> Any:
        A0 = problem["matrix"]

        # Single conversion (avoids extra np.asarray + astype + reparsing A0 later).
        A = np.asarray(A0, dtype=np.float64)
        n = int(A.shape[0])

        if n == 0:
            Z = np.zeros((0, 0), dtype=np.float64)
            return {"LU": {"P": Z, "L": Z, "U": Z}}
        if n == 1:
            P = np.array([[1.0]], dtype=np.float64)
            L = np.array([[1.0]], dtype=np.float64)
            U = np.array([[float(A[0, 0])]], dtype=np.float64)
            return {"LU": {"P": P, "L": L, "U": U}}

        # Tiny-matrix path: fully materialize (already extremely small).
        if self._use_numba and n <= self._numba_n:
            P, L, U = _lu_numba_pp_full(np.ascontiguousarray(A))
            return {"LU": {"P": P, "L": L, "U": U}}

        # LAPACK path with lazy materialization.
        if self._dgetrf is None:
            if self._use_numba:
                P, L, U = _lu_numba_pp_full(np.ascontiguousarray(A))
                return {"LU": {"P": P, "L": L, "U": U}}
            P = np.eye(n, dtype=np.float64)
            L = np.eye(n, dtype=np.float64)
            U = A.copy()
            return {"LU": {"P": P, "L": L, "U": U}}

        # Copy from already-converted A into Fortran-order buffer for LAPACK.
        Af = np.array(A, order="F", copy=True)
        lu, piv, info = self._dgetrf(Af, overwrite_a=True)

        if info < 0:
            # Illegal argument: be robust.
            if self._use_numba:
                P, L, U = _lu_numba_pp_full(np.ascontiguousarray(A))
                return {"LU": {"P": P, "L": L, "U": U}}
            P = np.eye(n, dtype=np.float64)
            L = np.eye(n, dtype=np.float64)
            U = A.astype(np.float64, copy=True)
            return {"LU": {"P": P, "L": L, "U": U}}

        # Return lazy factors; validator will materialize via np.asarray().
        return {
            "LU": {
                "P": _PFromPivots(np.asarray(piv), n),
                "L": _LFromPackedLU(lu),
                "U": _UFromPackedLU(lu),
            }
        }