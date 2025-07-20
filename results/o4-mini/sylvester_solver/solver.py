from scipy.linalg import solve_sylvester as _ss

class Solver:
    __slots__ = ()

    def solve(self, problem, **kwargs):
        A = problem['A']; B = problem['B']; Q = problem['Q']
        X = _ss(A, B, Q)
        return {'X': X}