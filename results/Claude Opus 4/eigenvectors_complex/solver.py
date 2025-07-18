import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """Compute eigenpairs of a square matrix and return sorted eigenvectors."""
        A = problem
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Use argsort for efficient sorting
        # Sort by real part (descending), then imaginary part (descending)
        sort_indices = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        
        sorted_evecs = []
        for idx in sort_indices:
            vec = eigenvectors[:, idx]
            # Fast normalization
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec = vec / norm
            sorted_evecs.append(vec.tolist())
        
        return sorted_evecs