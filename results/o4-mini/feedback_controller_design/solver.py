import numpy as np
from scipy.linalg import solve_discrete_are
from numpy.linalg import solve as _solve

_identity_cache = {}
def _eye(n):
    I = _identity_cache.get(n)
    if I is None:
        I = np.eye(n)
        _identity_cache[n] = I
    return I

class Solver:
    def solve(self, problem, **_):
        _dare = solve_discrete_are
        _solve_l = _solve
        _I = _eye
        try:
            A = np.asarray(problem['A'], dtype=float)
            B = np.asarray(problem['B'], dtype=float)
            n, m = A.shape[0], B.shape[1]
            I_n = _I(n); I_m = _I(m)
            P = _dare(A, B, I_n, I_m)
            PB = P.dot(B)
            S = I_m + PB.T.dot(B)
            K = -_solve_l(S, PB.T.dot(A))
            return {'is_stabilizable': True, 'K': K, 'P': P}
        except:
            return {'is_stabilizable': False, 'K': None, 'P': None}