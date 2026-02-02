import numpy as np
from scipy.linalg import svd as scipy_svd
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute SVD of matrix A using scipy with optimizations.
        """
        A = problem["matrix"]
        if isinstance(A, np.ndarray) and A.dtype == np.float64 and A.flags['F_CONTIGUOUS']:
            pass  # Already optimal
        elif isinstance(A, np.ndarray):
            A = np.asfortranarray(A, dtype=np.float64)
        else:
            A = np.array(A, dtype=np.float64, order='F')

        # Use scipy with gesdd driver and overwrite_a for speed
        U, s, Vh = scipy_svd(A, full_matrices=False, lapack_driver='gesdd',
                             check_finite=False, overwrite_a=True)

        return {"U": U, "S": s, "V": Vh.T}