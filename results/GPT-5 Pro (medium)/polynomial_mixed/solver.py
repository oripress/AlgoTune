from __future__ import annotations

from typing import Any
import cmath

import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the polynomial root-finding problem.

        Input: list/array-like of real coefficients [a_n, ..., a_0] (descending powers)
        Output: list of complex roots sorted by real part (desc), then imaginary part (desc)
        """
        # Convert to numpy array of float for np.roots; early exits for trivial cases
        coeffs = np.asarray(problem, dtype=np.float64)

        # Handle empty input
        if coeffs.size == 0:
            return []

        # Remove leading zeros (highest-degree) to avoid issues
        coeffs = np.trim_zeros(coeffs, trim="f")
        n = coeffs.size
        if n <= 1:
            # Constant polynomial (including all-zeros after trim): no roots
            return []

        # Factor out trailing zeros (roots at x=0) to reduce problem size for root finding
        k_trailing = 0
        # Manual scan to avoid extra allocations
        for i in range(n - 1, -1, -1):
            if coeffs[i] == 0.0:
                k_trailing += 1
            else:
                break

        if k_trailing > 0:
            nz_coeffs = coeffs[:-k_trailing]
        else:
            nz_coeffs = coeffs

        # If after removing trailing zeros only a single coefficient remains,
        # the polynomial is a*x^m with all m roots at 0.
        if nz_coeffs.size <= 1:
            roots = np.zeros(k_trailing, dtype=np.complex128)
        else:
            deg = nz_coeffs.size - 1
            # Fast paths for small degrees
            if deg == 1:
                # a1*x + a0 = 0 -> root = -a0/a1
                a1, a0 = nz_coeffs
                r = -a0 / a1
                roots = np.array([complex(r)], dtype=np.complex128)
            elif deg == 2:
                # a2*x^2 + a1*x + a0 = 0; use numerically stable quadratic formula
                a2, a1, a0 = nz_coeffs
                # Discriminant (may be negative), use cmath for complex sqrt
                disc = a1 * a1 - 4.0 * a2 * a0
                sqrt_disc = cmath.sqrt(disc)
                # Use sign trick to avoid catastrophic cancellation
                if a1 >= 0:
                    q = -0.5 * (a1 + sqrt_disc)
                else:
                    q = -0.5 * (a1 - sqrt_disc)
                # In degenerate cases where q is 0 (rare after trailing-zero removal), fall back
                if q == 0:
                    r1 = (-a1 + sqrt_disc) / (2.0 * a2)
                    r2 = (-a1 - sqrt_disc) / (2.0 * a2)
                else:
                    r1 = q / a2
                    r2 = a0 / q
                roots = np.array([complex(r1), complex(r2)], dtype=np.complex128)
            else:
                # General case: use numpy's optimized companion-matrix eigenvalues
                roots = np.roots(nz_coeffs)

        # Append trailing zero roots if any
        if k_trailing > 0:
            if isinstance(roots, np.ndarray):
                zeros = np.zeros(k_trailing, dtype=np.complex128)
                roots = np.concatenate((roots, zeros))
            else:
                # Roots might be a Python list in some unlikely path; normalize to ndarray
                roots = np.asarray(roots, dtype=np.complex128)
                zeros = np.zeros(k_trailing, dtype=np.complex128)
                roots = np.concatenate((roots, zeros))

        # Sort descending by real, then imaginary parts using numpy (faster than Python sorted)
        # np.lexsort sorts by last key primary; we want primary = real desc, secondary = imag desc
        idx = np.lexsort((-roots.imag, -roots.real))
        sorted_roots = roots[idx]

        # Return as Python list of complex numbers
        return [complex(z) for z in sorted_roots]