from typing import Any
import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[np.ndarray]:
        A = problem["matrix"]
        k = problem["k"]
        N = A.shape[0]
        
        # Create deterministic starting vector matching reference implementation
        v0 = np.ones(N, dtype=A.dtype)
        
        # Use reference implementation parameters for stability
        eigenvalues, eigenvectors = sparse.linalg.eigs(
            A,
            k=k,
            v0=v0,
            maxiter=N * 200,
            ncv=max(2*k + 1, 20),
            which='LM'  # Explicitly specify largest magnitude
        )
        
        # Optimized vectorized sorting using argsort
        abs_vals = np.abs(eigenvalues)
        idx = np.argsort(-abs_vals, kind='stable')
        solution = [eigenvectors[:, i] for i in idx]
        
        return solution