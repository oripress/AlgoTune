import numpy as np
import cmath
from typing import Any, List, Sequence

class Solver:
    def solve(self, problem: Sequence[float], **kwargs) -> List[complex]:
        """
        Compute all roots of a polynomial with real coefficients.

        :param problem: sequence of coefficients [a_n, a_{n-1}, ..., a_0]
                        representing p(x) = a_n x^n + ... + a_0
        :return: list of complex roots sorted by real part (desc), then imaginary part (desc)
        """
        if problem is None:
            return []

        coeffs = np.asarray(problem)
        # Ensure 1-D
        coeffs = coeffs.reshape(-1)

        if coeffs.size == 0:
            return []

        # Ensure numeric float conversion (inputs are real by spec)
        try:
            coeffs = coeffs.astype(float)
        except Exception:
            coeffs = np.array([float(x) for x in coeffs], dtype=float)

        # Remove leading zeros to determine true degree
        nz = np.flatnonzero(np.abs(coeffs) > 0)
        if nz.size == 0:
            # All-zero polynomial: no finite roots
            return []
        coeffs = coeffs[nz[0]:]
        n = coeffs.size

        # Constant polynomial -> no roots
        if n <= 1:
            return []

        # Linear: a x + b = 0
        if n == 2:
            a = float(coeffs[0]); b = float(coeffs[1])
            if a == 0.0:
                # Degenerate: fallback to numpy for consistent behavior
                computed = np.roots(coeffs)
                roots = [complex(r) for r in computed]
                roots.sort(key=lambda z: (z.real, z.imag), reverse=True)
                return roots
            return [complex(-b / a)]

        # Quadratic: use numerically stable quadratic formula supporting complex results
        if n == 3:
            a = complex(coeffs[0]); b = complex(coeffs[1]); c = complex(coeffs[2])
            if a == 0:
                # Degenerate to linear
                if b == 0:
                    return []
                return [complex(-c / b)]
            disc = b * b - 4.0 * a * c
            sqrt_disc = cmath.sqrt(disc)
            # Stable computation to reduce cancellation
            if b.real >= 0:
                q = -0.5 * (b + sqrt_disc)
            else:
                q = -0.5 * (b - sqrt_disc)
            if q != 0:
                r1 = q / a
                r2 = c / q
            else:
                r1 = (-b + sqrt_disc) / (2.0 * a)
                r2 = (-b - sqrt_disc) / (2.0 * a)
            roots = [complex(r1), complex(r2)]
            roots.sort(key=lambda z: (z.real, z.imag), reverse=True)
            return roots

        # Higher-degree polynomials: delegate to numpy.roots
        computed = np.roots(coeffs)
        roots = [complex(r) for r in computed]
        roots.sort(key=lambda z: (z.real, z.imag), reverse=True)
        return roots