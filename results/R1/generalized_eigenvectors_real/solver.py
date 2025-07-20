import numpy as np
from scipy.linalg import eigh

class Solver:
    def solve(self, problem, **kwargs):
        """Highly optimized generalized eigenvalue solver."""
        A, B = problem if isinstance(problem, tuple) and len(problem) == 2 else (problem[0], problem[1])
        
        # Create contiguous float32 arrays
        A_arr = np.asarray(A, dtype=np.float32, order='C')
        B_arr = np.asarray(B, dtype=np.float32, order='C')
        
        # Compute eigenvalues/eigenvectors with optimal settings
        eigenvalues, eigenvectors = eigh(
            A_arr, B_arr,
            driver='gvd',  # Fastest divide-and-conquer method
            overwrite_a=True,
            overwrite_b=True,
            check_finite=False
        )
        
        # Reverse in-place for efficiency
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        # Optimized conversion: use efficient transpose and memory layout
        eigenvectors_t = eigenvectors.T.copy()  # Efficient transpose with contiguous memory
        
        return eigenvalues.tolist(), eigenvectors_t.tolist()