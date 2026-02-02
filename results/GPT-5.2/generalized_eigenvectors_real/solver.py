from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np

try:
    import scipy.linalg as _sla
    from scipy.linalg.blas import get_blas_funcs as _get_blas_funcs
except Exception:  # pragma: no cover
    _sla = None
    _get_blas_funcs = None

class Solver:
    """
    Symmetric generalized eigenproblem A x = lambda B x with B SPD.

    Performance-focused:
      - n=1,2: specialized fast paths.
      - otherwise: Cholesky reduction B=LL^T, Atilde=L^{-1} A L^{-T},
        standard symmetric eigh, backtransform x=L^{-T} q.

    When SciPy is available, use BLAS TRSM for blocked triangular solves
    (supports right-side solve, avoiding transpose/copies).
    """

    def __init__(self) -> None:
        self._np = np

        npl = np.linalg
        self._np_eigh = npl.eigh
        self._np_chol = npl.cholesky
        self._np_solve = npl.solve

        if _sla is not None:
            self._sp_chol = _sla.cholesky
            self._sp_solve_tri = _sla.solve_triangular
        else:  # pragma: no cover
            self._sp_chol = None
            self._sp_solve_tri = None

        # BLAS TRSM (triangular solve with multiple RHS), if available
        self._trsm: Callable[..., np.ndarray] | None = None
        if _get_blas_funcs is not None and _sla is not None:
            try:
                # dtype inference ensures we get the float64 variant
                self._trsm = _get_blas_funcs("trsm", dtype=np.float64)
            except Exception:  # pragma: no cover
                self._trsm = None

    def solve(self, problem: Tuple[np.ndarray, np.ndarray], **kwargs) -> Any:
        A_in, B_in = problem

        asarray = np.asarray
        f64 = np.float64

        A = asarray(A_in, dtype=f64)
        B = asarray(B_in, dtype=f64)
        n = A.shape[0]

        if n == 1:
            b = float(B[0, 0])
            lam = float(A[0, 0]) / b
            return ([lam], [[1.0 / np.sqrt(b)]])

        if n == 2:
            # Closed-form solution of det(A - λB)=0, plus B-normalization.
            a11 = float(A[0, 0])
            a12 = float(A[0, 1])
            a22 = float(A[1, 1])

            b11 = float(B[0, 0])
            b12 = float(B[0, 1])
            b22 = float(B[1, 1])

            c2 = b11 * b22 - b12 * b12
            d1 = a11 * b22 + a22 * b11 - 2.0 * a12 * b12
            c0 = a11 * a22 - a12 * a12

            disc = d1 * d1 - 4.0 * c2 * c0
            if disc < 0.0:
                disc = 0.0
            sdisc = float(np.sqrt(disc))
            inv2c2 = 0.5 / c2
            l1 = (d1 + sdisc) * inv2c2
            l2 = (d1 - sdisc) * inv2c2

            def vec_for(lam: float) -> list[float]:
                m11 = a11 - lam * b11
                m12 = a12 - lam * b12
                m22 = a22 - lam * b22

                v0 = -m12
                v1 = m11
                if abs(v0) + abs(v1) < 1e-18:
                    v0 = m22
                    v1 = -m12
                    if abs(v0) + abs(v1) < 1e-18:
                        v0, v1 = 1.0, 0.0

                bn = b11 * v0 * v0 + 2.0 * b12 * v0 * v1 + b22 * v1 * v1
                invn = 1.0 / float(np.sqrt(bn))
                return [v0 * invn, v1 * invn]

            if l1 >= l2:
                return ([l1, l2], [vec_for(l1), vec_for(l2)])
            return ([l2, l1], [vec_for(l2), vec_for(l1)])

        # General case: Cholesky reduction + symmetric eigh
        if self._sp_chol is not None:
            try:
                L = self._sp_chol(B, lower=True, check_finite=False, overwrite_a=False)

                trsm = self._trsm
                if trsm is not None:
                    # tmp = L^{-1} A   (left solve)
                    tmp = trsm(
                        1.0,
                        L,
                        A,
                        side=0,  # left
                        lower=1,
                        trans_a=0,
                        diag=0,
                        overwrite_b=0,
                    )
                    # Atilde = tmp * L^{-T}   (right solve, avoids transpose)
                    Atilde = trsm(
                        1.0,
                        L,
                        tmp,
                        side=1,  # right
                        lower=1,
                        trans_a=1,  # inv(L^T)
                        diag=0,
                        overwrite_b=1,
                    )
                    w, q = self._np_eigh(Atilde)
                    # x = L^{-T} q
                    x = trsm(
                        1.0,
                        L,
                        q,
                        side=0,  # left
                        lower=1,
                        trans_a=1,
                        diag=0,
                        overwrite_b=0,
                    )
                else:
                    solve_tri = self._sp_solve_tri
                    tmp = solve_tri(L, A, lower=True, trans="N", check_finite=False, overwrite_b=False)
                    Atilde = solve_tri(
                        L, tmp.T, lower=True, trans="N", check_finite=False, overwrite_b=False
                    ).T
                    w, q = self._np_eigh(Atilde)
                    x = solve_tri(L, q, lower=True, trans="T", check_finite=False, overwrite_b=False)

                w = w[::-1]
                x = x[:, ::-1]
                return (w.tolist(), x.T.tolist())
            except Exception:
                pass

        # NumPy fallback
        L = self._np_chol(B)
        tmp = self._np_solve(L, A)
        Atilde = self._np_solve(L, tmp.T).T
        w, q = self._np_eigh(Atilde)
        x = self._np_solve(L.T, q)

        w = w[::-1]
        x = x[:, ::-1]
        return (w.tolist(), x.T.tolist())