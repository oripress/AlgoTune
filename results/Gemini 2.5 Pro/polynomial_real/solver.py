import numpy as np
from typing import Any, List

class Solver:
    def solve(self, problem: List[float], **kwargs) -> Any:
        """
        Finds all real roots of a polynomial using the companion matrix method
        implemented in numpy.roots. This approach is robust and standard for
        this problem.
        """
        coeffs = np.array(problem, dtype=np.float64)

        # A polynomial of degree < 1 has no roots.
        if len(coeffs) <= 1:
            return []

        # Trim leading zeros for numerical stability.
        first_nonzero = np.argmax(np.abs(coeffs) > 1e-15)
        coeffs = coeffs[first_nonzero:]

        # If trimming results in a constant, there are no roots.
        if len(coeffs) <= 1:
            return []

        # np.roots finds all complex roots of the polynomial.
        roots = np.roots(coeffs)
        
        # The problem guarantees all roots are real. Any imaginary part found
        # is due to numerical floating-point errors. Therefore, we take the
        # real part of all computed roots.
        real_roots = np.real(roots)
        
        # The problem requires the roots to be sorted in descending order.
        real_roots = np.sort(real_roots)[::-1]
        
        return real_roots.tolist()