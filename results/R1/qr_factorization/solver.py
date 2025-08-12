import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """Optimal QR factorization using NumPy's LAPACK backend."""
        A = problem["matrix"]
        Q, R = np.linalg.qr(A, mode='reduced')
        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}