import numpy as np
from scipy.linalg import svd as scipy_svd

class Solver:
    def solve(self, problem, **kwargs):
        A = np.array(problem["matrix"], dtype=np.float64)
        U, s, Vh = scipy_svd(A, full_matrices=False, check_finite=False, lapack_driver='gesdd')
        V = Vh.T
        return {"U": U, "S": s, "V": V}