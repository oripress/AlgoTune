import numpy as np
from scipy.linalg import eig
from typing import Any

class Solver:
    def solve(self, problem: list[float], **kwargs) -> Any:
        """
        Solve the polynomial problem by finding all roots.
        This solver uses a hybrid approach for speed and robustness:
        1. The entire fast-path logic is wrapped in a try-except block.
        2. If any error occurs, it falls back to the robust numpy.roots.
        3. The fast path pre-processes coefficients by trimming leading and
           trailing zeros (handling roots at zero efficiently).
        4. It uses fast, analytical formulas for degree 1 and 2 polynomials.
        5. For higher degrees, it uses scipy.linalg.eig on the standard
           companion matrix, which can be faster than numpy.roots.
        """
        try:
            coefficients = np.array(problem, dtype=np.float64)
            
            # Trim leading zeros
            first_nonzero_idx = np.flatnonzero(coefficients)
            if not first_nonzero_idx.size:
                return []
            
            coefficients = coefficients[first_nonzero_idx[0]:]

            # Handle roots at zero by trimming trailing zeros
            num_trailing_zeros = 0
            if len(coefficients) > 1:
                last_nonzero_idx = np.flatnonzero(coefficients[::-1])
                if last_nonzero_idx.size > 0:
                    num_trailing_zeros = last_nonzero_idx[0]
                    if num_trailing_zeros > 0:
                        coefficients = coefficients[:-num_trailing_zeros]

            degree = len(coefficients) - 1

            if degree <= 0:
                return [0.0j] * num_trailing_zeros

            if degree == 1:
                a, b = coefficients
                roots = np.array([-b / a], dtype=np.complex128)
            elif degree == 2:
                a, b, c = coefficients
                delta_sqrt = np.lib.scimath.sqrt(b**2 - 4*a*c)
                if b >= 0:
                    r1 = (-b - delta_sqrt) / (2*a)
                else:
                    r1 = (-b + delta_sqrt) / (2*a)
                r2 = c / (a * r1) if not np.isclose(r1, 0) else -b/a
                roots = np.array([r1, r2])
            else:
                monic_coeffs = coefficients / coefficients[0]
                A = np.zeros((degree, degree), dtype=np.float64)
                A[0, :] = -monic_coeffs[1:]
                np.fill_diagonal(A[1:, :], 1.0)
                roots = eig(A, left=False, right=False, check_finite=False)

            if num_trailing_zeros > 0:
                all_roots = np.concatenate((roots, np.zeros(num_trailing_zeros)))
            else:
                all_roots = roots
            
            order = np.lexsort((-all_roots.imag, -all_roots.real))
            return all_roots[order].tolist()

        except Exception:
            coeffs = np.array(problem, dtype=np.float64)
            computed_roots = np.roots(coeffs)
            order = np.lexsort((-computed_roots.imag, -computed_roots.real))
            return computed_roots[order].tolist()