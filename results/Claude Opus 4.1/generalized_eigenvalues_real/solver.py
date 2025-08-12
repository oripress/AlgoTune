import numpy as np
from scipy import linalg
from numpy.typing import NDArray
from typing import Any

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray], **kwargs) -> Any:
        """
        Solve the generalized eigenvalue problem for the given matrices A and B.
        
        The problem is defined as: A · x = λ B · x.
        Using scipy.linalg.eigh which is optimized for symmetric generalized eigenvalue problems.
        """
        A, B = problem
        
        # Use scipy's optimized generalized eigenvalue solver
        # This returns eigenvalues in ascending order
        eigenvalues = linalg.eigh(A, B, eigvals_only=True)
        
        # scipy.linalg.eigh returns in ascending order, so just reverse
        # This is much faster than sorting
        return eigenvalues[::-1].tolist()