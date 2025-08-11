from __future__ import annotations

from typing import Any, List

import cmath
import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray, **kwargs: Any) -> List[complex]:
        """
        Compute eigenvalues of a real square matrix and return them sorted in descending order:
        first by real part (descending), then by imaginary part (descending).
        """
        A = np.asarray(problem)
        n = A.shape[0]

        # Small sizes: closed forms
        if n == 1:
            return [complex(A[0, 0])]
        if n == 2:
            a, b = A[0, 0], A[0, 1]
            c, d = A[1, 0], A[1, 1]
            tr = a + d
            det = a * d - b * c
            disc = tr * tr - 4.0 * det  # (a+d)^2 - 4(ad - bc)
            sqrt_disc = cmath.sqrt(complex(disc))
            lam1 = (tr + sqrt_disc) / 2.0
            lam2 = (tr - sqrt_disc) / 2.0
            vals_2x2 = [complex(lam1), complex(lam2)]
            vals_2x2.sort(key=lambda z: (-z.real, -z.imag))
            return vals_2x2

        # Fast path for exactly symmetric real matrices
        if A.dtype.kind != "c" and np.array_equal(A, A.T):
            vals = np.linalg.eigvalsh(A)[::-1]
            return vals.tolist()

        # General case
        vals = np.linalg.eigvals(A)
        # Sort by real desc, then imag desc using numpy's complex sort + reverse
        sorted_vals = np.sort_complex(vals)[::-1]

        # Return Python complex numbers
        return sorted_vals.tolist()