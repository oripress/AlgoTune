import numpy as np
from typing import Any, Sequence

class Solver:
    def solve(self, problem: Sequence[float], **kwargs) -> Any:
        """
        Compute roots of a polynomial given coefficients in descending order.

        Returns a list of real roots sorted in descending order.
        Converts near-real complex roots to real when the imaginary part is small (tol=1e-3).
        Handles edge cases like empty input, all-zero coefficients, and constant polynomials.
        """
        if problem is None:
            return []

        try:
            coeffs = np.asarray(problem, dtype=float)
        except Exception:
            return []

        if coeffs.size == 0:
            return []

        # Remove leading zeros (highest-degree coefficients). If all zeros, no roots.
        coeffs = np.trim_zeros(coeffs, 'f')
        if coeffs.size == 0:
            return []

        # Constant polynomial has no roots.
        if coeffs.size == 1:
            return []

        # Use numpy.roots for general robust root finding.
        roots = np.roots(coeffs)

        # Convert roots to real when the imaginary parts are negligible (tol=1e-3)
        roots = np.real_if_close(roots, tol=1e-3)
        roots = np.real(roots)

        # Filter non-finite entries and sort in descending order.
        roots = roots[np.isfinite(roots)]
        if roots.size == 0:
            return []

        roots = np.sort(roots)[::-1]
        return roots.tolist()