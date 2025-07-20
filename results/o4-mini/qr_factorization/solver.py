import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        # Convert input to Fortran-order array for efficient LAPACK usage
        A = np.array(problem["matrix"], dtype=np.float64, order='F')
        # Perform reduced QR factorization
        Q, R = np.linalg.qr(A, mode='reduced')
        # Return numpy arrays directly to avoid Python-level list conversion overhead
        return {"QR": {"Q": Q, "R": R}}