import numpy as np
from functools import lru_cache

class Solver:
    @staticmethod
    @lru_cache(maxsize=None)
    def _cached_roots(coeffs: tuple) -> tuple:
        """Compute roots for a given coefficient tuple using NumPy."""
        roots = np.roots(coeffs)
        roots = np.real_if_close(roots, tol=1e-3)
        roots = np.real(roots)
        roots = np.sort(roots)[::-1]
        return tuple(roots.tolist())

    def solve(self, problem, **kwargs):
        """
        Compute all real roots of a polynomial with real coefficients.
        The polynomial is given by its coefficients in descending order.
        Returns the roots sorted in decreasing order as a list of floats.
        """
        # Ensure coefficients are floats
        coeffs = tuple(float(c) for c in problem)

        # Trim insignificant leading zeros (they do not affect the polynomial)
        coeffs = tuple(c for c in coeffs if not (abs(c) < 1e-12 and len(coeffs) > 1 and coeffs.index(c) == 0))
        # If all coefficients were zero (zero polynomial) or only a constant term remains, there are no roots
        if len(coeffs) <= 1:
            return []

        degree = len(coeffs) - 1

        # Linear case a*x + b = 0
        if degree == 1:
            a, b = coeffs
            if abs(a) < 1e-12:
                return []
            return [-b / a]

        # Quadratic case a*x^2 + b*x + c = 0
        if degree == 2:
            a, b, c = coeffs
            if abs(a) < 1e-12:
                # Degenerate to linear
                if abs(b) < 1e-12:
                    return []
                return [-c / b]
            disc = b * b - 4.0 * a * c
        # Higher degree – compute roots via companion matrix for speed
        degree = len(coeffs) - 1
        a0 = coeffs[0]
        coeffs_norm = np.array(coeffs, dtype=float) / a0
        C = np.zeros((degree, degree), dtype=float)
        C[0, :] = -coeffs_norm[1:]
        if degree > 1:
            C[1:, :-1] = np.eye(degree - 1)
        eigs = np.linalg.eigvals(C)
        roots = np.real_if_close(eigs, tol=1e-3)
        roots = np.real(roots)
        roots = np.sort(roots)[::-1]
        return roots.tolist()
        return list(self._cached_roots(coeffs))