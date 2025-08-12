import numpy as np
import numba as nb

@nb.njit(fastmath=True, cache=True)
def outer_product_numba(vec1, vec2):
    n1 = len(vec1)
    n2 = len(vec2)
    result = np.empty((n1, n2), dtype=vec1.dtype)
    for i in range(n1):
        for j in range(n2):
            result[i, j] = vec1[i] * vec2[j]
    return result

class Solver:
    def solve(self, problem, **kwargs):
        """Calculate the outer product of two vectors using Numba."""
        vec1, vec2 = problem
        vec1 = np.asarray(vec1, dtype=np.float64)
        vec2 = np.asarray(vec2, dtype=np.float64)
        return outer_product_numba(vec1, vec2)