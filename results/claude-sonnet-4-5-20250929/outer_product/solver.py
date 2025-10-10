import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        vec1, vec2 = problem
        return np.outer(vec1, vec2)