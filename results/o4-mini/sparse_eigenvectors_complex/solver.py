import numpy as np
from scipy.sparse.linalg import eigs

class Solver:
    def solve(self, problem, **kwargs):
        A = problem["matrix"]
        k = problem["k"]
        N = A.shape[0]
        # deterministic start vector
        v0 = np.ones(N, dtype=A.dtype)
        # compute eigenvalues and eigenvectors using ARPACK
        eigenvalues, eigenvectors = eigs(
            A,
            k=k,
            v0=v0,
            maxiter=N * 200,
            ncv=max(2 * k + 1, 20),
        )
        # pair eigenvalues with corresponding eigenvectors and sort by magnitude
        pairs = list(zip(eigenvalues, eigenvectors.T))
        pairs.sort(key=lambda pair: -np.abs(pair[0]))
        # return sorted eigenvectors
        return [pair[1] for pair in pairs]