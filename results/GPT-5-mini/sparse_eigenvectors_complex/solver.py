from typing import Any, List
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[np.ndarray]:
        """
        Compute the eigenvectors corresponding to the k eigenvalues with largest modulus.

        Returns:
            A list of k numpy arrays (complex) containing the eigenvectors sorted in
            descending order by the modulus of their eigenvalues.
        """
        A_in = problem["matrix"]
        k = int(problem.get("k", 0))

        if k <= 0:
            return []

        # Convert plain Python lists to numpy arrays; keep scipy sparse as-is.
        A = A_in
        if not sparse.issparse(A):
            A = np.asarray(A)

        # Ensure square matrix
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Input matrix must be square.")

        N = A.shape[0]

        # Deterministic starting vector using matrix dtype (match reference)
        dtype = getattr(A, "dtype", None)
        if dtype is None:
            dtype = np.complex128
        try:
            v0 = np.ones(N, dtype=dtype)
        except Exception:
            v0 = np.ones(N, dtype=np.complex128)

        # If requesting all (or more) eigenvalues, use dense solver
        if k >= N:
            if sparse.issparse(A):
                A_dense = A.toarray()
            else:
                A_dense = np.asarray(A)
            vals, vecs = np.linalg.eig(A_dense)
            pairs = list(zip(vals, vecs.T))
            pairs.sort(key=lambda pair: -np.abs(pair[0]))  # sort by descending magnitude
            solution = [pair[1] for pair in pairs][:k]
            return solution

        # ARPACK parameters matching the reference implementation
        ncv = max(2 * k + 1, 20)
        maxiter = N * 200

        # Compute k eigenpairs with largest magnitude (deterministic start vector)
        vals, vecs = spla.eigs(A, k=k, v0=v0, maxiter=maxiter, ncv=ncv)

        # Sort eigenvectors by descending modulus of eigenvalue and return them
        pairs = list(zip(vals, vecs.T))
        pairs.sort(key=lambda pair: -np.abs(pair[0]))
        solution = [pair[1] for pair in pairs][:k]
        return solution