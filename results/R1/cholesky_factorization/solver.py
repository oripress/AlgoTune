import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        A = problem["matrix"]
        L = np.linalg.cholesky(A)
        solution = {"Cholesky": {"L": L.tolist()}}
        return solution