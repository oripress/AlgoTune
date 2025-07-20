import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the polynomial problem by finding all real roots of the polynomial.
        The polynomial is given as [a_n, ..., a_0].
        Uses numpy.roots and returns real parts (tol=1e-3), sorted descending.
        """
        coeffs = problem
        roots = np.roots(coeffs)
        roots = np.real_if_close(roots, tol=1e-3)
        roots = np.real(roots)
        roots = np.sort(roots)[::-1]
        return roots.tolist()