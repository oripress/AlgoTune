import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        # Convert to numpy array
        A = np.array(problem, dtype=np.float64)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Precompute real and imaginary parts
        real_parts = eigenvalues.real
        imag_parts = eigenvalues.imag
        
        # Use lexsort with correct key order (last key is primary)
        # Primary: descending real, Secondary: descending imag
        indices = np.lexsort((-imag_parts, -real_parts))
        
        # Sort eigenvectors using advanced indexing
        sorted_eigenvectors = eigenvectors[:, indices]
        
        # Vectorized normalization
        norms = np.linalg.norm(sorted_eigenvectors, axis=0)
        non_zero = norms > 1e-12
        sorted_eigenvectors[:, non_zero] /= norms[non_zero]
        
        # Convert to list of lists (each column is an eigenvector)
        return sorted_eigenvectors.T.tolist()
        # Convert to numpy arrays for efficient conversion to lists
        sorted_eigenvectors_np = np.array(sorted_eigenvectors)
        
        # Efficient conversion to list of lists
        return sorted_eigenvectors_np.T.tolist()