import numpy as _np
from numpy import asarray as _a
from scipy.linalg import svd as _svd

class Solver:
    def solve(self, p, **k):
        A = _a(p["A"], dtype=_np.float64)
        B = _a(p["B"], dtype=_np.float64)
        M = B.dot(A.T)
        U, _, Vt = _svd(M, full_matrices=False, lapack_driver='gesdd')
        return {"solution": U.dot(Vt)}