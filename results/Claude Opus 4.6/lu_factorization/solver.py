import numpy as np
from scipy.linalg import lu
from scipy.linalg.decomp_lu import get_lapack_funcs as _get_lapack_funcs

class Solver:
    def __init__(self):
        # Warm up scipy.linalg.lu to avoid first-call overhead
        dummy = np.array([[1.0, 0.0], [0.0, 1.0]])
        lu(dummy)
    
    def solve(self, problem, **kwargs):
        A = problem["matrix"]
        if not isinstance(A, np.ndarray):
            A = np.asarray(A, dtype=np.float64)
        elif A.dtype != np.float64:
            A = A.astype(np.float64)
        P, L, U = lu(A, overwrite_a=True, check_finite=False)
        return {"LU": {"P": P, "L": L, "U": U}}