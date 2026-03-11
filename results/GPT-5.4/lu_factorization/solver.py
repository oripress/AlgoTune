import numpy as np
from scipy.linalg import lu

class Solver:
    def solve(self, problem, **kwargs):
        matrix = np.asarray(problem["matrix"], dtype=np.float64)
        p_mat, l_mat, u_mat = lu(matrix, check_finite=False)
        return {"LU": {"P": p_mat, "L": l_mat, "U": u_mat}}