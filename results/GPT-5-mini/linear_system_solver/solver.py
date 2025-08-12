from typing import Any, List
import numpy as np

class Solver:
    def solve(self, problem: Any, **kwargs) -> List[float]:
        """
        Solve the linear system A x = b.

        Expects `problem` to be a dict with keys "A" (n x n list) and "b" (length-n list).
        Uses np.linalg.solve for square systems and falls back to np.linalg.lstsq
        for non-square or singular cases. Returns the solution as a list of floats.
        """
        if not isinstance(problem, dict):
            raise ValueError("problem must be a dict with keys 'A' and 'b'")
        if "A" not in problem or "b" not in problem:
            raise ValueError("problem must contain keys 'A' and 'b'")

        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        # Handle trivial empty case
        if A.size == 0 and b.size == 0:
            return []

        # Ensure A is 2-D (treat 1-D as a single row)
        if A.ndim == 1:
            A = A.reshape(1, -1)

        # Flatten b to 1-D vector
        b = np.asarray(b).reshape(-1)

        # If shapes don't align but transposed A would match, try transpose (robustness)
        if A.ndim == 2 and A.shape[0] != b.shape[0] and A.shape[1] == b.shape[0]:
            A = A.T

        try:
            # Prefer the direct solver for square systems
            if A.ndim == 2 and A.shape[0] == A.shape[1] and A.shape[0] == b.shape[0]:
                x = np.linalg.solve(A, b)
            else:
                x, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback to least-squares on failure
            x, *_ = np.linalg.lstsq(A, b, rcond=None)

        x = np.asarray(x).reshape(-1)

        if x.size != 0 and not np.all(np.isfinite(x)):
            raise ValueError("Solution contains non-finite values")

        return [float(v) for v in x.tolist()]