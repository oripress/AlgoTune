import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> np.ndarray:
        vec1, vec2 = problem
        # Direct outer product using NumPy's optimized implementation
        return np.outer(vec1, vec2)