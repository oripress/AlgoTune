import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
        """
        Compute the matrix exponential exp(A).
        """
        A = np.asarray(problem["matrix"])
        expA = expm(A)
        return {"exponential": expA.tolist()}