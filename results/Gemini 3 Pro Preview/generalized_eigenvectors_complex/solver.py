import numpy as np
import scipy.linalg as la

class Solver:
    def solve(self, problem, **kwargs):
        A, B = problem
        
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        
        # Scale matrices for better numerical stability (matching reference)
        # Using Frobenius norm
        scale_B = np.sqrt(np.linalg.norm(B))
        if scale_B == 0:
            scale_B = 1.0
            
        inv_scale = 1.0 / scale_B
        
        # Create scaled copies
        A_scaled = A * inv_scale
        B_scaled = B * inv_scale
        
        # Solve generalized eigenvalue problem
        # overwrite_a and overwrite_b allow reusing memory of scaled matrices
        # check_finite=False skips NaN/Inf checks for speed
        eigenvalues, eigenvectors = la.eig(A_scaled, B_scaled, overwrite_a=True, overwrite_b=True, check_finite=False)
        
        # Normalize eigenvectors to unit Euclidean norm
        norms = np.linalg.norm(eigenvectors, axis=0)
        
        # Avoid division by zero and only normalize vectors with significant norm
        mask = norms > 1e-15
        if np.any(mask):
            eigenvectors[:, mask] /= norms[mask]
            
        # Sort eigenvalues and eigenvectors
        # Sort by real part descending, then imaginary part descending
        # lexsort sorts by keys in order (last key is primary)
        # We sort ascending by (real, imag) and then reverse
        order = np.lexsort((eigenvalues.imag, eigenvalues.real))
        order = order[::-1]
        
        sorted_eigenvalues = eigenvalues[order]
        sorted_eigenvectors = eigenvectors[:, order]
        
        # Convert to lists
        # sorted_eigenvectors is (n, n) with columns as eigenvectors
        # We need a list of lists where each inner list is an eigenvector
        # Transpose to get rows (which correspond to columns of original)
        return sorted_eigenvalues.tolist(), sorted_eigenvectors.T.tolist()