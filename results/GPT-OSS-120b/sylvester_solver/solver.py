from typing import Any, Dict
import numpy as np
from scipy.linalg import solve_sylvester

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the Sylvester equation A X + X B = Q for X.

        Parameters
        ----------
        problem : dict
            Dictionary containing matrices 'A', 'B', 'Q'.

        Returns
        -------
        dict
            Dictionary with key 'X' containing the solution matrix.
        """
        # Convert inputs to NumPy arrays (complex)
        A = np.asarray(problem["A"], dtype=np.complex128)
        B = np.asarray(problem["B"], dtype=np.complex128)
        Q = np.asarray(problem["Q"], dtype=np.complex128)

        try:
            # Fast eigen‑decomposition based solution
            lamA, VA = np.linalg.eig(A)
            lamB, VB = np.linalg.eig(B)

            # Solve VA^{-1} @ Q
            M = np.linalg.solve(VA, Q)          # (n, m)

            # Multiply by VB on the right
            N = M @ VB                          # (n, m)

            # Element‑wise division by (λ_i + μ_j)
            denom = lamA[:, None] + lamB[None, :]
            Y = N / denom

            # Form X = VA @ Y @ VB^{-1}
            X = VA @ np.linalg.solve(VB.T, Y.T).T
        except Exception:
            # Fallback to robust SciPy implementation
            X = solve_sylvester(A, B, Q)
        return {"X": X}