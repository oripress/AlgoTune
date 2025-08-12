import numpy as np
from numba import jit

@jit(nopython=True)
def outer_product_numba(vec1, vec2):
    n = len(vec1)
    result = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            result[i, j] = vec1[i] * vec2[j]
    return result

class Solver:
    def solve(self, problem, **kwargs) -> np.ndarray:
        """Calculate the outer product of two vectors."""
        vec1, vec2 = problem
        return outer_product_numba(vec1, vec2)