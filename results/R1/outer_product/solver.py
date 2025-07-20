import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        vec1, vec2 = problem
        return vec1.reshape(-1, 1) * vec2