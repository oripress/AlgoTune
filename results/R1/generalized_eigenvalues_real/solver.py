import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem, **kwargs):
        A, B = problem
        n = len(A)
        
        # Convert to Fortran-ordered arrays for optimal performance
        A_arr = np.array(A, dtype=np.float64, order='F')
        B_arr = np.array(B, dtype=np.float64, order='F')
        
        # Choose fastest driver based on matrix size
        driver = 'gvd' if n <= 200 else 'gvx'
        
        # Compute eigenvalues with optimization flags
        eigenvalues = scipy.linalg.eigh(
            A_arr, B_arr,
            overwrite_a=True,
            overwrite_b=True,
            check_finite=False,
            driver=driver,
            eigvals_only=True
        )
        
        # Return eigenvalues sorted in descending order
        return eigenvalues[::-1].tolist()