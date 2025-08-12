from __future__ import annotations

from typing import Any, Iterable, List
import numpy as np

class Solver:
    def solve(self, problem: Iterable[float], **kwargs: Any) -> List[complex]:
        """
        Solve the polynomial by finding all its roots.

        The input is a sequence of real coefficients in descending order:
            p(x) = a_n x^n + a_{n-1} x^{n-1} + ... + a_0

        Returns a list of roots (complex numbers), sorted in descending order by real part,
        and then by imaginary part.
        """
        # Convert to numpy array (avoid copying if already an array)
        coeffs = np.asarray(problem)

        # Identify non-zero coefficient range (leading and trailing zeros) efficiently
        mask = coeffs != 0
        if not np.any(mask):
            # All zeros -> numpy.roots would return empty array
            return []

        start = int(np.argmax(mask))
        end = len(mask) - int(np.argmax(mask[::-1])) - 1
        core = coeffs[start : end + 1]
        # Trailing zeros count corresponds to multiplicity of root at x=0
        t = len(coeffs) - 1 - end

        deg = core.size - 1
        if deg <= 0:
            roots = np.empty(0, dtype=np.complex128)
        elif deg == 1:
            # Linear ax + b = 0
            a, b = core
            roots = np.array([-b / a], dtype=np.complex128)
        elif deg == 2:
            # Quadratic ax^2 + bx + c = 0; numerically stable quadratic formula
            a, b, c = core
            disc = b * b - 4.0 * a * c
            sqrt_disc = np.sqrt(disc + 0.0j)
            if b >= 0:
                q = -0.5 * (b + sqrt_disc)
            else:
                q = -0.5 * (b - sqrt_disc)
            if q == 0:
                r = -b / (2.0 * a)
                roots = np.array([r, r], dtype=np.complex128)
            else:
                r1 = q / a
                r2 = c / q
                roots = np.array([r1, r2], dtype=np.complex128)
        else:
            roots = np.roots(core)

        # Append zero roots if any trailing zeros were present
        if t > 0:
            roots = np.concatenate((roots, np.zeros(t, dtype=np.complex128)), axis=0)

        # Sort by real part then imaginary part, both descending (avoid allocating negatives)
        if roots.size > 1:
            idx = np.lexsort((roots.imag, roots.real))  # ascending by real then imag
            roots = roots[idx[::-1]]  # reverse for descending

        # Return as a list of Python complex numbers
        return roots.tolist()