from typing import Any, Dict
import numpy as np


class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        A = problem["A"]
        B = problem["B"]
        Q = problem["Q"]
        try:
            A = np.asarray(A)
            B = np.asarray(B)
            Q = np.asarray(Q)
            wA, VA = np.linalg.eig(A)
            wB, VB = np.linalg.eig(B)
            C = np.linalg.solve(VA, Q) @ VB
            Y = C / (wA[:, None] + wB[None, :])
            VAY = VA @ Y
            X = np.linalg.solve(VB.T, VAY.T).T
            return {"X": X}
        except Exception:
            from scipy.linalg import solve_sylvester
            return {"X": solve_sylvester(A, B, Q)}