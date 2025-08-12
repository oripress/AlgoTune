from typing import Any, List
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs, ArpackNoConvergence

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[np.ndarray]:
        """
        Compute the k eigenvectors with largest-magnitude eigenvalues of a sparse matrix.
        Uses a hybrid approach: dense solver for small matrices, 
        fast ARPACK calls with limited iterations for large matrices,
        and fallbacks ensuring correct results.
        """
        A = problem["matrix"]
        k = problem["k"]
        N = A.shape[0]
        # deterministic start vector
        v0 = np.ones(N, dtype=A.dtype)

        # Use dense solver for small matrices for speed and reliability
        if N <= 100:
            M = A.toarray() if sparse.issparse(A) else np.asarray(A)
            all_vals, all_vecs = np.linalg.eig(M)
            idx = np.argsort(-np.abs(all_vals))[:k]
            return [all_vecs[:, i] for i in idx]

        # Use sparse ARPACK solver for large matrices
        try:
            vals, vecs = eigs(
                A,
                k=k,
                v0=v0,
                maxiter=N * 200,
                ncv=max(2 * k + 1, 20),
                which='LM'
            )
            if vals.size < k:
                raise ArpackNoConvergence(eigenvalues=vals, eigenvectors=vecs)
        except ArpackNoConvergence:
            # fallback to dense solver
            M = A.toarray() if sparse.issparse(A) else np.asarray(A)
            all_vals, all_vecs = np.linalg.eig(M)
            idx = np.argsort(-np.abs(all_vals))[:k]
            return [all_vecs[:, i] for i in idx]

        # sort eigenpairs by descending magnitude and return eigenvectors
        pairs = list(zip(vals, vecs.T))
        pairs.sort(key=lambda pair: -np.abs(pair[0]))
        return [pair[1] for pair in pairs]