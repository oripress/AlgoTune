from __future__ import annotations

import cmath
from typing import Any, List

import numpy as np

def _cbrt(z: complex) -> complex:
    """Principal complex cube root."""
    r, phi = cmath.polar(z)
    return cmath.rect(r ** (1.0 / 3.0), phi / 3.0)

def _sort_roots(roots: Any) -> List[complex]:
    r = np.asarray(roots, dtype=np.complex128)
    if r.size <= 1:
        return r.tolist()
    # Descending by real part, then descending by imaginary part.
    idx = np.lexsort((-r.imag, -r.real))
    return r[idx].tolist()

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        coeffs = np.asarray(problem, dtype=np.float64)

        # Trim leading zeros (match np.roots behavior of reducing degree).
        if coeffs.size == 0:
            return []
        nz = np.flatnonzero(coeffs)
        if nz.size == 0:
            return []
        if nz[0] != 0:
            coeffs = coeffs[nz[0] :]

        n = int(coeffs.size - 1)
        if n <= 0:
            return []

        # Fast paths for small degrees (real coefficients).
        if n == 1:
            a0, a1 = float(coeffs[0]), float(coeffs[1])
            if a0 == 0.0 or not np.isfinite(a0):
                return []
            return [complex(-a1 / a0, 0.0)]

        if n == 2:
            a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
            if a == 0.0:
                # Degenerate to linear.
                if b == 0.0:
                    return []
                return [complex(-c / b, 0.0)]
            disc = complex(b * b - 4.0 * a * c, 0.0)
            s = cmath.sqrt(disc)
            inv2a = 0.5 / a
            r1 = (-b + s) * inv2a
            r2 = (-b - s) * inv2a
            return _sort_roots((r1, r2))

        if n == 3:
            a, b, c, d = map(float, coeffs[:4])
            if a == 0.0:
                # Degenerate to quadratic.
                return self.solve([b, c, d])

            inva = 1.0 / a
            bb = b * inva
            cc = c * inva
            dd = d * inva

            # Depressed cubic: y^3 + p y + q = 0 with x = y - bb/3
            shift = bb / 3.0
            p = cc - (bb * bb) / 3.0
            q = (2.0 * bb * bb * bb) / 27.0 - (bb * cc) / 3.0 + dd

            half_q = -0.5 * q
            delta = complex(half_q * half_q + (p / 3.0) * (p / 3.0) * (p / 3.0), 0.0)
            sqrt_delta = cmath.sqrt(delta)

            u = _cbrt(half_q + sqrt_delta)
            v = _cbrt(half_q - sqrt_delta)

            y0 = u + v
            omega = complex(-0.5, 0.86602540378443864676)  # exp(2πi/3)
            omega2 = omega * omega
            y1 = u * omega + v * omega2
            y2 = u * omega2 + v * omega

            x0 = y0 - shift
            x1 = y1 - shift
            x2 = y2 - shift
            return _sort_roots((x0, x1, x2))

        # General case: rely on NumPy's optimized eigenvalue-based routine.
        roots = np.roots(coeffs)
        return _sort_roots(roots)