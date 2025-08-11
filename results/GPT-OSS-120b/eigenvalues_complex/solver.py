import numpy as np
from typing import Any, List

class Solver:
    def solve(self, problem: Any, **kwargs) -> List[complex]:
        """
        Compute eigenvalues of a real square matrix and return them sorted
        in descending order by real part, then by imaginary part.
        """
        # Convert input to a NumPy array (handles list of lists or ndarray)
        if isinstance(problem, np.ndarray):
            matrix = problem.astype(np.float64, copy=False)
        else:
            matrix = np.asarray(problem, dtype=np.float64)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(matrix)

        # Sort eigenvalues in descending order by real part then imag part
        # Using NumPy's sort_complex (ascending by real then imag) and reverse
        sorted_vals = np.sort_complex(eigenvalues)[::-1]
        # Return as a list of Python complex numbers
        return sorted_vals.tolist()