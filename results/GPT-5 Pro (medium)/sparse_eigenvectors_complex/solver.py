from typing import Any, List

import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[np.ndarray]:
        """
        Compute the eigenvectors corresponding to the k eigenvalues of largest magnitude
        for a given (potentially sparse) square matrix.

        Returns a list of eigenvectors sorted in descending order by the modulus of their
        corresponding eigenvalues. Matches the reference implementation's parameters.
        """
        A = problem["matrix"]
        k = int(problem["k"])
        n = A.shape[0]

        # Deterministic starting vector using matrix dtype
        v0 = np.ones(n, dtype=A.dtype)

        # Use ARPACK via scipy.sparse.linalg.eigs with same params as reference
        vals, vecs = sparse.linalg.eigs(
            A,
            k=k,
            v0=v0,
            maxiter=n * 200,
            ncv=max(2 * k + 1, 20),
        )

        # Stable sort by descending modulus without constructing Python pair objects
        keys = -np.abs(vals)
        idx = np.argsort(keys, kind="mergesort")  # stable ascending on -|val| == stable descending on |val|
        solution = [vecs[:, i] for i in idx]
        return solution