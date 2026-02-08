import numpy as np
from scipy.linalg import qz as _scipy_qz

class Solver:
    def __init__(self):
        # Pre-warm
        a = np.eye(2, dtype=np.float64)
        b = np.eye(2, dtype=np.float64)
        _scipy_qz(a, b, output='real')

    def solve(self, problem, **kwargs):
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        
        AA, BB, Q, Z = _scipy_qz(A, B, output='real')
        
        return {"QZ": {"AA": AA, "BB": BB, "Q": Q, "Z": Z}}