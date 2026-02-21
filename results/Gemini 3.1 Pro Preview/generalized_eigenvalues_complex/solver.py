import scipy.linalg as la
import numpy as np
import torch
import torch.linalg as tla
from typing import Any

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray], **kwargs) -> Any:
        A, B = problem
        try:
            # Convert to PyTorch tensors
            A_t = torch.from_numpy(A)
            B_t = torch.from_numpy(B)
            
            # Solve B * X = A
            C_t = tla.solve(B_t, A_t)  # pylint: disable=not-callable
            
            # Compute eigenvalues
            eigenvalues_t = tla.eigvals(C_t)  # pylint: disable=not-callable
            eigenvalues = eigenvalues_t.numpy()
        except Exception:
            # Fallback to generalized eigenvalue solver if B is singular or solve fails
            eigenvalues = la.eigvals(A, B, check_finite=False)
            
        # Sort eigenvalues: descending order by real part, then by imaginary part
        sorted_eigenvalues = -np.sort(-eigenvalues)
        return sorted_eigenvalues.tolist()