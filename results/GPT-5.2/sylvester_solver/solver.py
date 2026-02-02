from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import schur, solve_sylvester
from scipy.linalg.blas import get_blas_funcs
from scipy.linalg.lapack import get_lapack_funcs

try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover
    threadpool_limits = None

def _schur_cf(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Complex Schur with check_finite disabled when available."""
    try:
        return schur(A, output="complex", overwrite_a=True, check_finite=False)
    except TypeError:
        return schur(A, output="complex", overwrite_a=True)

# Cache BLAS/LAPACK kernels for complex128 (init-time cost doesn't count).
_d128 = np.empty((1, 1), dtype=np.complex128)
_ZGEMM = get_blas_funcs(("gemm",), (_d128,))[0]
_ZTRSYL = get_lapack_funcs(("trsyl",), (_d128,))[0]

class Solver:
    # Enter threadpool limit once (process-wide) to avoid per-call overhead.
    _tp_cm: Optional[Any] = None
    _tp_entered: bool = False

    def __init__(self) -> None:
        if threadpool_limits is not None and not Solver._tp_entered:
            cm = threadpool_limits(limits=1)
            cm.__enter__()
            Solver._tp_cm = cm
            Solver._tp_entered = True

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        A = np.asarray(problem["A"])
        B = np.asarray(problem["B"])
        Q = np.asarray(problem["Q"])

        try:
            if A.dtype != np.complex128:
                A = A.astype(np.complex128, copy=False)
            if B.dtype != np.complex128:
                B = B.astype(np.complex128, copy=False)
            if Q.dtype != np.complex128:
                Q = Q.astype(np.complex128, copy=False)

            T, U = _schur_cf(A)
            S, V = _schur_cf(B)

            # C = U^H @ Q @ V
            QV = _ZGEMM(alpha=1.0, a=Q, b=V, trans_a=0, trans_b=0)
            C = _ZGEMM(alpha=1.0, a=U, b=QV, trans_a=2, trans_b=0)

            Y, scale, info = _ZTRSYL(T, S, C, trana="N", tranb="N", isgn=1, overwrite_c=1)
            if int(info) != 0:
                raise RuntimeError("trsyl failed")
            if scale not in (0.0, 1.0):
                Y = Y / scale

            UY = _ZGEMM(alpha=1.0, a=U, b=Y, trans_a=0, trans_b=0)
            X = _ZGEMM(alpha=1.0, a=UY, b=V, trans_a=0, trans_b=2)
            return {"X": X}
        except Exception:
            return {"X": solve_sylvester(A, B, Q)}