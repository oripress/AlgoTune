from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        """
        Solve the linear system Ax = b using NumPy's optimized solver.

        Args:
            problem (dict): Dictionary with keys "A" (square matrix) and "b" (vector).

        Returns:
            list[float]: Solution vector x such that A x = b.
        """
        # Convert inputs to numpy arrays with minimal overhead
        A = np.asarray(problem["A"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        # Solve without forming the inverse for speed and stability
        x = np.linalg.solve(A, b)

        # Return as a plain Python list
        return x.tolist()