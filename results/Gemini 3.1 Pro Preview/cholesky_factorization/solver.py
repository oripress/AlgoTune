import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem: dict, **kwargs):
        A = np.array(problem["matrix"], dtype=np.float64, order='F')
        L = scipy.linalg.cholesky(A, lower=True, overwrite_a=True, check_finite=False)
        return {"Cholesky": {"L": L}}