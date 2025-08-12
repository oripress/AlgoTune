import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        A = np.asarray(problem["matrix"], dtype=np.float64)
        Q, R = np.linalg.qr(A, mode='reduced')
        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}