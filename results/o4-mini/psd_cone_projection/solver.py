import numpy as np
from typing import Any, Dict
from scipy.linalg.lapack import get_lapack_funcs

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Prepare matrix in Fortran order for LAPACK efficiency
        A = np.array(problem["A"], dtype=np.float64, order='F', copy=True)
        # Eigen-decomposition (symmetric) via LAPACK 'syevd' if available
        try:
            syevd, = get_lapack_funcs(('syevd',), (A,))
            w, v, info = syevd(A, compute_v=1, overwrite_a=1)
            if info != 0:
                raise RuntimeError("syevd failed")
        except Exception:
            w, v = np.linalg.eigh(A)
        # Clamp negatives to zero
        w[w < 0.0] = 0.0
        # Reconstruct PSD projection: V diag(w) V^T via broadcasting
        X = (v * w) @ v.T
        return {"X": X}