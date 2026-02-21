import numpy as np
import scipy.linalg as la
from typing import Any

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray], **kwargs) -> Any:
        A, B = problem
        
        # Try to convert to a standard eigenvalue problem for speed
        try:
            # B @ C = A  =>  C = B^-1 A
            C = np.linalg.solve(B, A)
            eigenvalues, eigenvectors = np.linalg.eig(C)
        except Exception:
            # Fallback to generalized solver if B is singular or ill-conditioned
            eigenvalues, eigenvectors = la.eig(A, B, overwrite_a=True, overwrite_b=True, check_finite=False)
        # Normalize eigenvectors (columns)
        norms = np.linalg.norm(eigenvectors, axis=0)
        norms[norms <= 1e-15] = 1.0
        eigenvectors /= norms
        
        # Sort by descending real part, then descending imaginary part
        # np.argsort on complex numbers sorts by real, then imaginary.
        # By negating, we sort in descending order for both.
        idx = np.argsort(-eigenvalues)
        
        sorted_eigenvalues = eigenvalues[idx].tolist()
        sorted_eigenvectors = eigenvectors[:, idx].T.tolist()
        
        return sorted_eigenvalues, sorted_eigenvectors