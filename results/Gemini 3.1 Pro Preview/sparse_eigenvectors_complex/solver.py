from typing import Any
import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[complex]:
        A = problem["matrix"]
        k = problem["k"]
        N = A.shape[0]
        v0 = np.ones(N, dtype=A.dtype)
        A_op = sparse.linalg.LinearOperator((N, N), matvec=A.dot, dtype=A.dtype)
        eigenvalues, eigenvectors = sparse.linalg.eigs(
            A_op,
            k=k,
            v0=v0,
            maxiter=N * 200,
            ncv=max(2 * k + 1, 20),
        )
        order = np.argsort(-np.abs(eigenvalues))
        solution = [eigenvectors[:, i] for i in order]
        return solution