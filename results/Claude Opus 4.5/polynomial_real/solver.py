import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        coefficients = problem
        computed_roots = np.roots(coefficients)
        computed_roots = np.real_if_close(computed_roots, tol=1e-3)
        computed_roots = np.real(computed_roots)
        computed_roots = np.sort(computed_roots)[::-1]
        return computed_roots