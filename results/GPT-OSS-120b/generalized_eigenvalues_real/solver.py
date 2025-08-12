import numpy as np
from typing import Any, Tuple, List

class Solver:
    def solve(self, problem: Tuple[np.ndarray, np.ndarray], **kwargs) -> List[float]:
        """
        Solve the generalized eigenvalue problem A·x = λ B·x for symmetric A
        and symmetric positive‑definite B.

        Parameters
        ----------
        problem : tuple (A, B)
            A – symmetric matrix (n×n)
            B – symmetric positive‑definite matrix (n×n)

        Returns
        -------
        List[float]
            Eigenvalues sorted in descending order.
        """
        A, B = problem

        # Ensure inputs are NumPy arrays (no copy if already ndarray)
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)

        # Cholesky decomposition of B (B = L Lᵀ)
        L = np.linalg.cholesky(B)

        # Compute A_tilde = L^{-1} @ A @ L^{-T} using explicit inverse (matches reference)
        Linv = np.linalg.inv(L)
        A_tilde = Linv @ A @ Linv.T
        # Compute eigenvalues of the transformed symmetric matrix
        eigvals = np.linalg.eigvalsh(A_tilde)

        # Return eigenvalues in descending order as plain Python floats
        return eigvals[::-1].astype(float).tolist()
        return sorted_vals
        return sorted_vals