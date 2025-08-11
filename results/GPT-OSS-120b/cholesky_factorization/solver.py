import numpy as np

class Solver:
    # Re‑usable result container to reduce allocation overhead
    _result_template = {"Cholesky": {"L": None}}

    def solve(self, problem, **kwargs):
        """
        Compute the Cholesky factorization of a symmetric positive definite matrix.
        """
        # Convert to a NumPy array (default float64) to retain full precision
        A = np.asarray(problem["matrix"])
        # Local reference to the LAPACK routine
        chol = np.linalg.cholesky
        L = chol(A)

        # Populate reusable result container
        inner = self._result_template["Cholesky"]
        inner["L"] = L
        return self._result_template