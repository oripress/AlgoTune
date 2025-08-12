from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.linalg import eigh as scipy_eigh

    _SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    scipy_eigh = None
    _SCIPY_AVAILABLE = False

class Solver:
    def solve(self, problem: Tuple[NDArray, NDArray], **kwargs) -> list[float]:
        """
        Solve the generalized eigenvalue problem A x = λ B x for symmetric A and SPD B.

        Fast path:
          - Use SciPy's generalized-symmetric eigenvalue solver (scipy.linalg.eigh)
            with eigvals_only=True and driver='gv' for speed.
        Fallback:
          - Replicate the reference approach via Cholesky, explicit inverse, and eigvalsh.
        """
        A, B = problem
        # Ensure float64 arrays; Fortran order can benefit LAPACK backends
        A = np.asarray(A, dtype=np.float64, order="F")
        B = np.asarray(B, dtype=np.float64, order="F")

        if _SCIPY_AVAILABLE:
            # Compute eigenvalues only; ascending order by default
            evals = scipy_eigh(
                A,
                B,
                lower=True,
                eigvals_only=True,
                check_finite=False,
                overwrite_a=True,
                overwrite_b=True,
                driver="gv",
            )
            return evals[::-1].tolist()

        # NumPy fallback: transform to standard eigenproblem
        L = np.linalg.cholesky(B)
        Linv = np.linalg.inv(L)
        Atilde = Linv @ A
        Atilde = Atilde @ Linv.T

        eigenvalues = np.linalg.eigvalsh(Atilde)
        return eigenvalues[::-1].tolist()