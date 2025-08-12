import numpy as np
from scipy.linalg import qz, get_lapack_funcs

class Solver:
    def solve(self, problem: dict, **kwargs):
        """
        Compute the QZ (generalized Schur) factorization of the matrix pair (A, B).

        Parameters
        ----------
        problem : dict
            Dictionary with keys "A" and "B" containing square matrices as nested lists.
        **kwargs : dict
            Additional keyword arguments (ignored).

        Returns
        -------
        dict
            Dictionary with key "QZ" mapping to another dictionary containing
            the factor matrices "AA", "BB", "Q", and "Z" as nested Python lists.
        """
        # Convert input lists to NumPy arrays (use float when possible)
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)

        # Perform QZ factorization using SciPy's implementation.
        # Use complex output (still valid) and disable extra checks for speed.
        AA, BB, Q, Z = qz(A, B, output="real",
                         overwrite_a=True, overwrite_b=True,
                         check_finite=False)

        # Convert results back to plain Python lists for the required output format.
        solution = {
            "QZ": {
                "AA": AA.tolist(),
                "BB": BB.tolist(),
                "Q": Q.tolist(),
                "Z": Z.tolist(),
            }
        }
        return solution