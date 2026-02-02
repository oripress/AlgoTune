from typing import Any
import numpy as np
from scipy import sparse
import scipy.linalg

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[complex]:
        A = problem["matrix"]
        k = problem["k"]
        N = A.shape[0]

        # Optimization: Use dense solver for small matrices or when k is large
        # Dense eig is roughly O(N^3). Sparse eigs is O(k * N * iter).
        # For small N, dense is faster and avoids ARPACK overhead.
        # If k is large relative to N, sparse solver iteration cost increases.
        if N < 400 or k > 0.2 * N:
            if sparse.issparse(A):
                A_dense = A.toarray()
            else:
                A_dense = A
            
            # scipy.linalg.eig returns (vals, vecs)
            # vecs[:, i] is the eigenvector corresponding to vals[i]
            # overwrite_a=True allows destroying A_dense for speed
            # check_finite=False skips NaN/Inf checks
            vals, vecs = scipy.linalg.eig(A_dense, overwrite_a=True, check_finite=False)
            
            # Sort by descending order of eigenvalue modulus
            # Use python sort for stability to match reference
            pairs = list(zip(vals, vecs.T))
            pairs.sort(key=lambda pair: -np.abs(pair[0]))
            
            solution = [pair[1] for pair in pairs[:k]]
            return solution

        # Create a deterministic starting vector
        v0 = np.ones(N, dtype=A.dtype)  # Use matrix dtype

        # Compute eigenvalues using sparse.linalg.eigs
        eigenvalues, eigenvectors = sparse.linalg.eigs(
            A,
            k=k,
            v0=v0,  # Add deterministic start vector
            maxiter=N * 200,
            ncv=max(2 * k + 1, 20),
        )

        pairs = list(zip(eigenvalues, eigenvectors.T))
        # Sort by descending order of eigenvalue modulus
        pairs.sort(key=lambda pair: -np.abs(pair[0]))

        solution = [pair[1] for pair in pairs]

        return solution