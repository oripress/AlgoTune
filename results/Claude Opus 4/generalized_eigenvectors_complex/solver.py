import numpy as np
import scipy.linalg as la

class Solver:
    def solve(self, problem):
        """
        Solve the generalized eigenvalue problem for the given matrices A and B:
            A · x = λ B · x.
        
        :param problem: Tuple (A, B) where A and B are n x n real matrices.
        :return: (eigenvalues, eigenvectors)
        """
        A, B = problem
        
        # Efficient Frobenius norm using einsum
        scale_B = np.sqrt(np.einsum('ij,ij->', B, B))
        
        # Scale matrices in-place to avoid copies
        inv_scale = 1.0 / scale_B
        A_scaled = A * inv_scale
        B_scaled = B * inv_scale
        
        # Solve scaled problem
        eigenvalues, eigenvectors = la.eig(A_scaled, B_scaled, overwrite_a=True, overwrite_b=True)
        
        # Normalize eigenvectors using einsum for efficiency
        # Calculate norms of each column
        norms = np.sqrt(np.einsum('ij,ij->j', np.real(eigenvectors), np.real(eigenvectors)) + 
                       np.einsum('ij,ij->j', np.imag(eigenvectors), np.imag(eigenvectors)))
        
        # Normalize only where norm > threshold
        valid_mask = norms > 1e-15
        eigenvectors[:, valid_mask] /= norms[valid_mask]
        
        # Efficient sorting: use lexsort directly
        # Note: lexsort sorts in ascending order, so we negate for descending
        sort_indices = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        
        # Apply sorting without creating intermediate arrays
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # Convert to lists efficiently
        # Use memoryview and direct conversion
        eigenvalues_list = eigenvalues.tolist()
        
        # For eigenvectors, transpose once and convert
        eigenvectors_T = eigenvectors.T
        eigenvectors_list = [vec.tolist() for vec in eigenvectors_T]
        
        return (eigenvalues_list, eigenvectors_list)