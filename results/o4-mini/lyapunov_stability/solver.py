import numpy as np
from numpy.linalg import eig, inv

class Solver:
    def solve(self, problem, **kwargs):
        A = np.asarray(problem.get("A"), dtype=float)
        # must be square
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return {"is_stable": False, "P": None}
        # eigen‐decomposition
        w, V = eig(A)
        # check spectral radius
        if np.abs(w).max() >= 1.0:
            return {"is_stable": False, "P": None}
        # invert eigenbasis
        W = inv(V)
        # form Q = V^T V
        Q = V.T @ V
        # denom matrix = 1 - w_i w_j
        denom = 1.0 - np.multiply.outer(w, w)
        # form X = Q / denom
        X = Q / denom
        # assemble P = W^T X W, symmetrize and real part
        P = W.T @ (X @ W)
        P = ((P + P.T) * 0.5).real
        return {"is_stable": True, "P": P}