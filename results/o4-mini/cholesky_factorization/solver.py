import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        A = problem["matrix"]
        L = np.linalg.cholesky(A)
        return {"Cholesky": {"L": L}}