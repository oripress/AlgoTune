import numpy as np
from scipy.linalg import qr

class Solver:
    def solve(self, problem, **kwargs):
        A = problem["matrix"]
        # Use scipy qr with optimizations
        if isinstance(A, np.ndarray):
            A = np.asfortranarray(A, dtype=np.float64)
        else:
            A = np.asarray(A, dtype=np.float64, order='F')
        Q, R = qr(A, mode='economic', check_finite=False, overwrite_a=True)
        return {"QR": {"Q": Q, "R": R}}