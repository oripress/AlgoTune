import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute C = A · B using NumPy BLAS-backed dot and return as list of lists.
        """
        # Load input matrices
        A = np.array(problem["A"], dtype=float)
        B = np.array(problem["B"], dtype=float)
        # Compute product
        C = A.dot(B)
        # Convert to list of row arrays for validation
        return list(C)