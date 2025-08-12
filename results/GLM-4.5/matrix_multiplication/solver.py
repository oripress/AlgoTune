import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """Try with np.dot and direct return without intermediate variable."""
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        return np.dot(A, B).tolist()