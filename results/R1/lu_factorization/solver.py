import numpy as np
from scipy.linalg import lu

class Solver:
    def solve(self, problem, **kwargs):
        A_list = problem["matrix"]
        n = len(A_list)
        A_arr = np.array(A_list, dtype=np.float64, order='C')
        
        # Use scipy's lu with optimization flags
        P, L, U = lu(A_arr, overwrite_a=True, check_finite=False)
        
        return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}