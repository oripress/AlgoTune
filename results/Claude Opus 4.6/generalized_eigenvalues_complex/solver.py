import numpy as np
import scipy.linalg as la
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        A, B = problem
        n = A.shape[0]
        
        if n == 0:
            return []
        
        # Scale matrices for better numerical stability.
        scale_B = np.sqrt(np.linalg.norm(B))
        if scale_B == 0:
            scale_B = 1.0
        B_scaled = B / scale_B
        A_scaled = A / scale_B
        
        # Use eigvals instead of eig to skip eigenvector computation
        eigenvalues = la.eigvals(A_scaled, B_scaled)
        
        # Sort eigenvalues: descending order by real part, then by imaginary part.
        idx = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        sorted_eigs = eigenvalues[idx]
        
        return sorted_eigs.tolist()