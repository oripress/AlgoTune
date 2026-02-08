import numpy as np
from scipy.linalg import eigh, cholesky, solve_triangular
from scipy.linalg import lapack

class Solver:
    def solve(self, problem, **kwargs):
        A, B = problem
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        n = A.shape[0]
        
        if n == 1:
            val = A[0, 0] / B[0, 0]
            vec = 1.0 / np.sqrt(B[0, 0])
            return ([float(val)], [[float(vec)]])
        
        # Use scipy eigh with check_finite=False and overwrite for speed
        eigenvalues, eigenvectors = eigh(A, B, check_finite=False, overwrite_a=True, overwrite_b=True)
        
        # Reverse to get descending order
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        # Convert to lists
        eigenvalues_list = eigenvalues.tolist()
        eigenvectors_list = eigenvectors.T.tolist()
        
        return (eigenvalues_list, eigenvectors_list)