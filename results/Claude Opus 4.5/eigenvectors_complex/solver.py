import numpy as np
from scipy import linalg

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute eigenvectors of a real square matrix, sorted by eigenvalues
        in descending order (by real part, then imaginary part).
        """
        A = np.ascontiguousarray(problem, dtype=np.float64)
        
        # Use scipy's eig with check_finite=False for speed
        eigenvalues, eigenvectors = linalg.eig(A, check_finite=False)
        
        # Sort indices by descending real part, then imaginary part
        idx = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        
        # Reorder and transpose to get eigenvectors as rows
        sorted_evecs = eigenvectors[:, idx].T
        
        # Ensure complex dtype
        if not np.iscomplexobj(sorted_evecs):
            sorted_evecs = sorted_evecs.astype(complex)
        
        # Normalize each row (eigenvector)
        norms = np.linalg.norm(sorted_evecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        sorted_evecs = sorted_evecs / norms
        
        return sorted_evecs.tolist()