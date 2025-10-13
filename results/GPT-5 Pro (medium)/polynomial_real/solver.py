from typing import Any, Iterable, List

import numpy as np
import math

class Solver:
    @staticmethod
    def _as_array(problem: Iterable[float]) -> np.ndarray:
        # Convert to numpy array without copying if possible
        arr = np.asarray(list(problem), dtype=float)
        # Remove exact leading zeros only (to match numpy.roots behavior on exact zeros)
        # Do not remove near-zeros to avoid altering problem definition.
        nz = np.flatnonzero(arr != 0.0)
        if nz.size == 0:
            return np.array([0.0])  # constant zero polynomial -> no roots (handled later)
        return arr[nz[0]:]

    @staticmethod
    def _linear_root(a: float, b: float) -> List[float]:
        # Solve a x + b = 0
        return [(-b) / a]

    @staticmethod
    def _quadratic_roots(a: float, b: float, c: float) -> List[float]:
        # Stable quadratic formula
        D = b * b - 4.0 * a * c
        if D < 0.0:
            # Shouldn't happen for real-rooted, but just fallback to complex sqrt for consistency
            sqrtD = math.sqrt(-D) * 1j
        else:
            sqrtD = math.sqrt(D)
        if b >= 0:
            q = -0.5 * (b + sqrtD)
        else:
            q = -0.5 * (b - sqrtD)
        if q == 0:
            # Double root or near-zero denominator
            r = -b / (2.0 * a)
            return [r, r]
        r1 = q / a
        r2 = c / q
        return [r1, r2]

    @staticmethod
    def _cubic_roots(a: float, b: float, c: float, d: float) -> List[float]:
        # Solve a x^3 + b x^2 + c x + d = 0
        if a == 0.0:
            # Degenerate to quadratic
            return Solver._quadratic_roots(b, c, d)

        # Convert to depressed cubic y^3 + p y + q = 0 via x = y - b/(3a)
        inv_a = 1.0 / a
        bb = b * b
        p = (3.0 * a * c - bb) * (inv_a * inv_a) / 3.0
        q = (2.0 * bb * b - 9.0 * a * b * c + 27.0 * (a * a) * d) * (inv_a * inv_a * inv_a) / 27.0

        # Discriminant
        half_q = 0.5 * q
        third_p = p / 3.0
        disc = half_q * half_q + third_p * third_p * third_p

        offset = b * inv_a / 3.0

        def cbrt(x: float) -> float:
            if x >= 0:
                return x ** (1.0 / 3.0)
            return -((-x) ** (1.0 / 3.0))

        if disc > 0.0:
            # One real root; for consistency with reference (which will output real parts for all),
            # we can still return three values by falling back to numpy to ensure matching.
            # However, typical benchmark assumes all real roots, so this path is rare.
            u = cbrt(-half_q + math.sqrt(disc))
            v = cbrt(-half_q - math.sqrt(disc))
            y1 = u + v
            # The other two roots are complex; return just real parts similar to reference:
            # But to avoid mismatch, fallback to numpy for this uncommon case.
            return None  # signal fallback
        elif disc == 0.0:
            # Multiple real roots: at least two are equal.
            if half_q == 0.0:
                y1 = 0.0
                roots = [y1, y1, y1]
            else:
                u = cbrt(-half_q)
                y1 = 2.0 * u
                y2 = -u
                roots = [y1, y2, y2]
        else:
            # Three distinct real roots
            # y_k = 2*sqrt(-p/3) * cos((phi + 2k*pi)/3), where cos(phi) = -q/(2) * ( -27/p^3 )^{-1/2}
            # Use numerically stable computation of phi.
            phi_arg = -half_q / math.sqrt(-(third_p * third_p * third_p))
            # Clamp to [-1, 1] to avoid numerical issues
            phi_arg = 1.0 if phi_arg > 1.0 else (-1.0 if phi_arg < -1.0 else phi_arg)
            phi = math.acos(phi_arg)
            t = 2.0 * math.sqrt(-third_p)
            y1 = t * math.cos(phi / 3.0)
            y2 = t * math.cos((phi + 2.0 * math.pi) / 3.0)
            y3 = t * math.cos((phi + 4.0 * math.pi) / 3.0)
            roots = [y1, y2, y3]

        # Convert back: x = y - b/(3a)
        shift = offset
        return [r - shift for r in roots]

    def solve(self, problem: Iterable[float], **kwargs: Any) -> List[float]:
        """
        Compute and return the real roots (real parts) of the polynomial defined by `problem`
        (coefficients in descending order), sorted in decreasing order.
        """
        coeffs = self._as_array(problem)
        n = coeffs.size - 1  # degree

        if n <= 0:
            return []

        # Direct, faster closed-form for small degrees
        if n == 1:
            a, b = coeffs
            roots = self._linear_root(a, b)
            roots.sort(reverse=True)
            return [float(r) for r in roots]
        if n == 2:
            a, b, c = coeffs
            roots = self._quadratic_roots(a, b, c)
            # Convert to real parts (for consistency with reference)
            roots = [float(np.real(r)) for r in roots]
            roots.sort(reverse=True)
            return roots
        if n == 3:
            a, b, c, d = coeffs
            roots = self._cubic_roots(a, b, c, d)
            if roots is not None:
                roots.sort(reverse=True)
                return [float(r) for r in roots]
            # Fallback to numpy for rare non-all-real cubic
            # Fall through to np.roots below.

        # General case: use numpy.roots and post-process like the reference
        computed_roots = np.roots(coeffs)
        computed_roots = np.real_if_close(computed_roots, tol=1e-3)
        computed_roots = np.real(computed_roots)
        computed_roots = np.sort(computed_roots)[::-1]
        return computed_roots.tolist()