import numpy as np
from typing import List
import math

class Solver:
    def solve(self, problem, **kwargs) -> List[float]:
        coeffs = problem
        n = len(coeffs)

        if n == 0:
            return []

        # Fast path for true small-degree polynomials with nonzero leading coefficient
        if n <= 3 and coeffs[0] != 0:
            if n == 1:
                # Constant polynomial: no roots
                roots = []
            elif n == 2:
                a, b = float(coeffs[0]), float(coeffs[1])
                roots = [-b / a]
            else:
                a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
                disc = b * b - 4.0 * a * c
                if disc >= 0.0:
                    sqrt_disc = math.sqrt(disc)
                    # Numerically stable quadratic formula
                    if b >= 0.0:
                        qv = -0.5 * (b + sqrt_disc)
                    else:
                        qv = -0.5 * (b - sqrt_disc)
                    if qv == 0.0:
                        r1 = (-b + sqrt_disc) / (2.0 * a)
                        r2 = (-b - sqrt_disc) / (2.0 * a)
                    else:
                        r1 = qv / a
                        r2 = c / qv
                    roots = [r1, r2]
                else:
                    # Complex conjugate pair -> both map to their common real part after post-processing
                    real_part = -b / (2.0 * a)
                    roots = [real_part, real_part]

            roots = sorted([float(r) for r in roots], reverse=True)
            return roots

        # General case: rely on numpy's robust implementation
        roots = np.roots(coeffs)
        roots = roots.real
        roots.sort()
        roots = roots[::-1]
        return roots.tolist()