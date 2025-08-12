import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> np.ndarray:
        """Calculate the outer product of two vectors."""
        vec1, vec2 = problem
        return np.multiply.outer(vec1, vec2)