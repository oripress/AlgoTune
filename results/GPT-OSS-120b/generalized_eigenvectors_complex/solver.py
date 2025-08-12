import numpy as np
import scipy.linalg as la
from typing import Any, List, Tuple

class Solver:
    def solve(self, problem: Tuple[np.ndarray, np.ndarray], **kwargs) -> Tuple[List[complex], List[List[complex]]]:
        """
        Solve the generalized eigenvalue problem A·x = λ·B·x for real matrices A and B.

        Returns eigenvalues sorted by descending real part then descending imaginary part,
        and the corresponding unit‑norm eigenvectors (as Python lists of complex numbers).
        """
        # Accept both tuple and list inputs for the problem.
        if isinstance(problem, (list, tuple)) and len(problem) == 2:
            A, B = problem
        else:
            raise ValueError("Problem must be a pair (A, B) of matrices.")

        # Scaling for numerical stability is omitted for speed; we operate directly on A and B.

        # Compute eigenvalues and right eigenvectors.
        # For invertible B we form C = B⁻¹·A and use NumPy's fast eig.
        # If B is singular, fall back to SciPy's generalized eig.
        try:
            C = np.linalg.solve(B, A)          # B⁻¹·A
            eigvals, eigvecs = np.linalg.eig(C)    # fast dense eigensolver
        except np.linalg.LinAlgError:
            eigvals, eigvecs = la.eig(A, B)   # fallback for singular B

        # Sort eigenvalues (descending real, then descending imag) using NumPy argsort for speed
        order = np.lexsort((-eigvals.imag, -eigvals.real))
        sorted_vals = eigvals[order]
        sorted_vecs = eigvecs[:, order]

        # Convert to plain Python structures
        eigenvalues = sorted_vals.tolist()
        eigenvectors = sorted_vecs.T.tolist()
        return eigenvalues, eigenvectors