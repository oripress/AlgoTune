import numpy as np
from typing import Any, List

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute eigenvalues for a real square matrix and return them sorted
        in descending order by real part, then by imaginary part.

        :param problem: list of lists or numpy array representing the square matrix
        :return: list of complex eigenvalues sorted by (-real, -imag)
        """
        a = np.asarray(problem)

        # Empty input -> no eigenvalues
        if a.size == 0:
            return []

        # Scalar input -> single value
        if a.ndim == 0:
            return [complex(a.item())]

        # 1-D input: accept only length-1 as a 1x1 matrix
        if a.ndim == 1:
            if a.shape[0] == 1:
                return [complex(a[0])]
            raise ValueError("Input must be a 2D square matrix.")

        # Ensure square matrix
        n, m = a.shape
        if n != m:
            raise ValueError("Input must be a square matrix.")

        # If numeric, ensure float64 for predictable performance
        if np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating):
            a = a.astype(np.float64, copy=False)

        # Compute eigenvalues (values only)
        vals = np.linalg.eigvals(a)

        # Sort by descending real part, then descending imaginary part.
        # Use numpy.lexsort for speed: last key is primary, so provide (-imag, -real)
        try:
            idx = np.lexsort((-vals.imag, -vals.real))
            sorted_vals = vals[idx]
        except Exception:
            # Fallback to Python sort if lexsort fails for any reason
            sorted_vals = np.array(sorted(vals, key=lambda x: (-x.real, -x.imag)))

        # Return as Python complex numbers
        return [complex(v) for v in sorted_vals]