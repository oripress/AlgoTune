from __future__ import annotations
from typing import Any, List
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[np.ndarray]:
        """
        Compute the eigenvectors corresponding to the `k` eigenvalues of largest magnitude
        of a given square sparse matrix.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
            - "matrix": a square scipy.sparse matrix (CSR format recommended)
            - "k": number of eigenvectors/eigenvalues to return

        Returns
        -------
        List[np.ndarray]
            List of `k` eigenvectors (as 1‑D complex NumPy arrays) sorted in descending order
            by the modulus of their associated eigenvalue.
        """
        A = problem["matrix"]
        k = problem["k"]

        # Ensure matrix is in CSR format for efficient arithmetic
        if not sparse.isspmatrix_csr(A):
            A = A.tocsr()

        n = A.shape[0]

        # Deterministic start vector (all ones) matching matrix dtype
        v0 = np.ones(n, dtype=A.dtype)

        # Use SciPy's eigs for sparse matrices.
        # The algorithm returns eigenvalues and eigenvectors (columns).
        # We request the k eigenvalues with largest magnitude.
        # For small matrices, fallback to dense eig if needed.
        try:
            eigenvalues, eigenvectors = eigs(
                A,
                k=k,
                which='LM',   # largest magnitude
                v0=v0,
                maxiter=n * 200,
                ncv=max(2 * k + 1, 20)
            )
        except Exception:
            # In rare cases (e.g., convergence issues), use dense eig.
            dense_A = A.toarray()
            eigenvalues, eigenvectors = np.linalg.eig(dense_A)
            # Select top‑k by magnitude
            idx = np.argsort(-np.abs(eigenvalues))[:k]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        # Pair eigenvalues with their eigenvectors (as 1‑D arrays)
        pairs = list(zip(eigenvalues, eigenvectors.T))

        # Sort by descending modulus of eigenvalue
        pairs.sort(key=lambda pair: -np.abs(pair[0]))

        # Extract sorted eigenvectors
        solution = [vec.astype(np.complex128) for _, vec in pairs]

        return solution