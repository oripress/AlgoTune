import numpy as np
from scipy import linalg

class Solver:
    def solve(self, problem, **kwargs):
        # Ensure input is a numpy array with proper dtype
        if not isinstance(problem, np.ndarray):
            problem = np.array(problem, dtype=np.float64)
        
        # Use scipy's eigvals which can be faster than numpy's eig
        # check_finite=False skips NaN/Inf checking for speed
        eigenvalues = linalg.eigvals(problem, check_finite=False)
        # Sort eigenvalues: descending order by real part, then by imaginary part
        # Use numpy's lexsort for potentially faster sorting
        indices = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        solution = eigenvalues[indices].tolist()
        return solution