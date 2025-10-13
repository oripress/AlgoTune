from typing import Any, Tuple, List
import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> Tuple[List[float], List[List[float]]]:
        """
        Compute eigenvalues and orthonormal eigenvectors of a real symmetric matrix.

        Returns:
          - eigenvalues: list of floats in descending order
          - eigenvectors: list of lists, where each inner list is the corresponding eigenvector,
            orthonormal and aligned with the sorted eigenvalues
        """
        # Ensure proper dtype and contiguous layout for performance
        A = np.asarray(problem, dtype=float)

        # eigh is specialized for symmetric (Hermitian) matrices; returns ascending eigenvalues
        w, V = np.linalg.eigh(A)

        # Reverse to descending order; V columns correspond to w entries
        w_desc = w[::-1]
        V_desc = V[:, ::-1]

        # Prepare outputs: rows as eigenvectors (each length n), consistent with validator expectations
        eigenvalues = w_desc.tolist()
        eigenvectors = V_desc.T.tolist()

        return eigenvalues, eigenvectors