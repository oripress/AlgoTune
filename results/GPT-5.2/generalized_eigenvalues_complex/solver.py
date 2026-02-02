from __future__ import annotations

from typing import Any

class Solver:
    """
    Fast generalized eigenvalues A x = λ B x.

    Common fast path (B invertible in typical random instances):
      eigvals(B^{-1} A) via one linear solve + standard eigvals (NumPy/LAPACK).

    Rare fallback:
      generalized eigvals via LAPACK dggev / SciPy if the solve fails.
    """

    def __init__(self) -> None:
        import numpy as np
        import scipy.linalg as la
        from scipy.linalg import lapack

        self.np = np
        self._solve = np.linalg.solve
        self._eigvals = np.linalg.eigvals
        self._lexsort = np.lexsort
        self._asfortran = np.asfortranarray

        self.la = la
        self.lapack = lapack
        self._have_dggev = hasattr(lapack, "dggev") and hasattr(lapack, "dggev_lwork")
        self._lwork_cache: dict[int, int] = {}

    def _generalized_eigvals(self, A, B):
        """Generalized eigenvalues via LAPACK dggev (no eigenvectors), with SciPy fallback."""
        if self._have_dggev:
            n = A.shape[0]
            lwork = self._lwork_cache.get(n)
            if lwork is None:
                lw, info = self.lapack.dggev_lwork(n, compute_vl=0, compute_vr=0)
                lwork = int(lw) if info == 0 else 0
                self._lwork_cache[n] = lwork

            alphar, alphai, beta, _, _, info = self.lapack.dggev(
                A,
                B,
                compute_vl=0,
                compute_vr=0,
                lwork=lwork if lwork > 0 else None,
                overwrite_a=1,
                overwrite_b=1,
            )
            if info == 0:
                return (alphar + 1j * alphai) / beta

        return self.la.eigvals(A, B, check_finite=False)

    def solve(self, problem, **kwargs) -> Any:
        np = self.np

        # In the benchmark, A and B are already numpy float64 ndarrays.
        A, B = problem

        try:
            eig = self._eigvals(self._solve(B, A))
        except Exception:
            # Fallback: need Fortran order for LAPACK, but avoid copies if already F-contiguous.
            if not getattr(A, "flags", None) or not A.flags.f_contiguous:
                A = self._asfortran(A)
            if not getattr(B, "flags", None) or not B.flags.f_contiguous:
                B = self._asfortran(B)
            eig = self._generalized_eigvals(A, B)

        order = self._lexsort((-eig.imag, -eig.real))
        return eig[order].tolist()