from typing import Any, Dict, List

import numpy as np
from scipy.linalg import lu

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Dict[str, List[List[float]]]]:
        # Use Fortran order to favor LAPACK routines, avoid unnecessary checks/copies
        A = np.asfortranarray(np.asarray(problem["matrix"], dtype=np.float64))
        n = A.shape[0]

        if n == 1:
            val = float(A[0, 0])
            return {"LU": {"P": [[1.0]], "L": [[1.0]], "U": [[val]]}}

        # SciPy's LU with minimal overhead
        P, L, U = lu(A, overwrite_a=True, check_finite=False)

        return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}