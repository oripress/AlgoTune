import numpy as np
from scipy.linalg import qz

class Solver:
    def solve(
        self, problem: dict[str, list[list[float]]]
    ) -> dict[str, dict[str, list[list[float | complex]]]]:
        """
        Solve the QZ factorization problem by computing the QZ factorization of (A,B).
        Uses scipy.linalg.qz with mode='real' to compute:
            A = Q AA Z*
            B = Q BB Z*
        """
        # Use Fortran-order (column-major) arrays for better LAPACK performance
        A = np.asfortranarray(problem["A"], dtype=np.float64)
        B = np.asfortranarray(problem["B"], dtype=np.float64)
        # Use overwrite flags to avoid unnecessary copies
        AA, BB, Q, Z = qz(A, B, output="real", overwrite_a=True, overwrite_b=True)
        return {"QZ": {"AA": AA.tolist(), "BB": BB.tolist(), "Q": Q.tolist(), "Z": Z.tolist()}}