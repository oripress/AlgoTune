import numpy as np
from scipy.linalg import eigh

class Solver:
    def solve(self, problem):
        # Load and center data
        X = np.asarray(problem["X"], dtype=np.float64)
        X -= X.mean(axis=0)
        m, n = X.shape
        # Number of components
        k = int(problem.get("n_components", 0))
        if k <= 0:
            return np.zeros((0, n), dtype=np.float64)
        if k > n:
            k = n

        # If features fewer than or equal to samples: eigen-decompose C = X^T X (n x n)
        if n <= m:
            C = np.empty((n, n), dtype=X.dtype)
            np.dot(X.T, X, out=C)
            dim = n
            # choose subset or full eigen solver
            if k * 2 < dim:
                eigvals, eigvecs = eigh(
                    C,
                    subset_by_index=[dim - k, dim - 1],
                    driver='evr',
                    overwrite_a=True,
                )
            else:
                eigvals, eigvecs = eigh(C, driver='evd', overwrite_a=True)
            # reverse order: largest first
            evecs = eigvecs[:, ::-1]
            # take top k eigenvectors as rows
            return evecs[:, :k].T

        # Otherwise eigen-decompose C2 = X X^T (m x m)
        C2 = np.empty((m, m), dtype=X.dtype)
        np.dot(X, X.T, out=C2)
        dim = m
        if k * 2 < dim:
            eigvals, u = eigh(
                C2,
                subset_by_index=[dim - k, dim - 1],
                driver='evr',
                overwrite_a=True,
            )
        else:
            eigvals, u = eigh(C2, driver='evd', overwrite_a=True)
        # reverse and select top k
        u = u[:, ::-1][:, :k]
        eigvals = eigvals[::-1][:k]
        # build principal directions: X^T u / sqrt(eigvals)
        V = X.T.dot(u) / np.sqrt(eigvals)
        # return as (k, n)
        return V.T