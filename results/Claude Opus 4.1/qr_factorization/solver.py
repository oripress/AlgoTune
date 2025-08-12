import numpy as np
from scipy.linalg import qr

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the QR factorization problem using scipy with optimized parameters.
        """
        # Convert to numpy array with C-contiguous layout for better performance
        A = np.asarray(problem["matrix"], dtype=np.float64, order='C')
        
        # Use scipy's QR with overwrite_a to avoid memory allocation
        Q, R = qr(A, mode='economic', overwrite_a=False, check_finite=False)
        
        solution = {"QR": {"Q": Q.tolist(), "R": R.tolist()}}
        return solution