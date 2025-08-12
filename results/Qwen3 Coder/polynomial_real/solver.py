import numpy as np
from typing import List, Any

class Solver:
    def solve(self, problem: List[float], **kwargs) -> Any:
        """
        Solve the polynomial problem by finding all real roots of the polynomial.
        """
        # Use numpy's polynomial roots function with direct computation
        computed_roots = np.roots(problem)
        
        # Convert to real numbers if imaginary parts are negligible
        real_values = np.real_if_close(computed_roots, tol=1e-3)
        real_values = np.real(real_values)
        
        # Use partitioning for faster sorting when we only need to sort
        indices = np.argsort(real_values)[::-1]
        return real_values[indices].tolist()