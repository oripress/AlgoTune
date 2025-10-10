import numpy as np
from scipy import linalg
from numba import jit

@jit(nopython=True, cache=True, fastmath=True)
def compute_procrustes(A, B):
    """Compute M = B @ A.T then SVD and return G."""
    M = np.dot(B, A.T)
    U, _, Vt = np.linalg.svd(M)
    G = np.dot(U, Vt)
    return G

class Solver:
    def solve(self, problem):
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)
        G = compute_procrustes(A, B)
        return {"solution": G.tolist()}