import numpy as np
from scipy.linalg import eigvals

class Solver:
    def solve(self, problem, **kwargs):
        # Compute eigenvalues using SciPy which is faster for larger matrices
        eigenvalues = eigvals(problem)
        
        # Precompute negative values for efficient sorting
        neg_real = -eigenvalues.real
        neg_imag = -eigenvalues.imag
        
        # Get sorted indices using tuple keys (more efficient than structured array)
        # lexsort sorts by last key first: primary key = neg_real, secondary key = neg_imag
        idx = np.lexsort((neg_imag, neg_real))
        return eigenvalues[idx].tolist()