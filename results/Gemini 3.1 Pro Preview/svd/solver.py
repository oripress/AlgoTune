import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem, **kwargs):
        A = np.array(problem["matrix"], dtype=np.float64, order='F')
        n, m = A.shape
        if n < m:
            U, s, Vh = scipy.linalg.svd(A.T, full_matrices=False, compute_uv=True, overwrite_a=True, check_finite=False, lapack_driver='gesdd')
            return {"U": Vh.T, "S": s, "V": U}
        else:
            U, s, Vh = scipy.linalg.svd(A, full_matrices=False, compute_uv=True, overwrite_a=True, check_finite=False, lapack_driver='gesdd')
            return {"U": U, "S": s, "V": Vh.T}