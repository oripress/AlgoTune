import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute all roots of a real‑coefficient polynomial given by `problem`,
        a list of coefficients in descending order, and return them sorted
        in descending order by real part, then by imaginary part.
        """
        if not problem:
            return np.array([], dtype=complex)
        degree = len(problem) - 1
        if degree == 0:
            return np.array([], dtype=complex)
        if degree == 1:
            a, b = problem
            return np.array([-b / a], dtype=complex)
        if degree == 2:
            a, b, c = problem
            disc = b * b - 4 * a * c
            sqrt_disc = np.sqrt(disc + 0j)
            r1 = (-b + sqrt_disc) / (2 * a)
            r2 = (-b - sqrt_disc) / (2 * a)
            roots = np.array([r1, r2], dtype=complex)
        else:
            roots = np.roots(problem)
        # Fast descending sort by real part then imaginary part
        idx = np.lexsort((-roots.imag, -roots.real))
        sorted_roots = roots[idx]
        return sorted_roots