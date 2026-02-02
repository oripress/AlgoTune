import numpy as np
from scipy.linalg import eigh
from scipy.linalg.lapack import dsygvd

class Solver:
    def solve(self, problem, **kwargs):
        A, B = problem
        n = A.shape[0]
        
        # Make copies for overwrite since we need contiguous arrays
        A_copy = np.asfortranarray(A, dtype=np.float64)
        B_copy = np.asfortranarray(B, dtype=np.float64)
        
        # Use LAPACK dsygvd directly (divide and conquer)
        # itype=1: A*v = lambda*B*v
        # jobz='V': compute eigenvalues and eigenvectors
        # uplo='L': use lower triangular part
        eigenvalues, eigenvectors, info = dsygvd(A_copy, B_copy, itype=1, jobz='V', uplo='L')[:3]
        
        # Reverse for descending order and convert to lists efficiently
        return (eigenvalues[::-1].tolist(), eigenvectors[:, ::-1].T.tolist())