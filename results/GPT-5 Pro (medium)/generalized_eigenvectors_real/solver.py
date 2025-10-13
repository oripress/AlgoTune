from __future__ import annotations

from typing import Any, Tuple, List

import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: Tuple[NDArray, NDArray], **kwargs) -> Tuple[List[float], List[List[float]]]:
        """
        Solve the generalized eigenvalue problem A x = λ B x for symmetric A and SPD B.

        Approach:
        - Compute Cholesky factorization B = L L^T.
        - Reduce to standard eigenproblem: Atilde = L^{-1} A L^{-T}.
          Do so with linear solves (no explicit inverses).
        - Solve Atilde u = λ u with np.linalg.eigh (symmetric).
        - Map eigenvectors back: x = L^{-T} u via triangular solve.
        - Optionally B-normalize eigenvectors for numerical robustness.
        - Return eigenvalues in descending order and corresponding eigenvectors.

        :param problem: Tuple (A, B) where A is symmetric and B is symmetric positive definite.
        :return: (eigenvalues_desc, eigenvectors_desc) with eigenvalues sorted descending and
                 eigenvectors B-normalized and corresponding to the returned eigenvalues.
        """
        A, B = problem
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        # Cholesky factorization: B = L L^T
        L = np.linalg.cholesky(B)

        # Form Atilde = L^{-1} A L^{-T} using linear solves to avoid explicit inverses.
        # First solve L X = A  -> X = L^{-1} A
        X = np.linalg.solve(L, A)
        # Then solve L^T Y^T = X^T  -> Y = X L^{-T}
        Atilde = np.linalg.solve(L.T, X.T).T

        # Solve the symmetric eigenvalue problem
        eigvals, U = np.linalg.eigh(Atilde)  # ascending order

        # Map eigenvectors back to original space: Xvecs = L^{-T} U
        # Solve L^T Xvecs = U
        Xvecs = np.linalg.solve(L.T, U)

        # Reverse order to descending eigenvalues and corresponding eigenvectors
        eigvals = eigvals[::-1]
        Xvecs = Xvecs[:, ::-1]

        # B-normalize eigenvectors for robustness (should already be B-orthonormal theoretically)
        # Compute B @ Xvecs once, then column-wise norms as sqrt(sum(X * (B@X), axis=0))
        BX = B @ Xvecs
        norms = np.sqrt(np.sum(Xvecs * BX, axis=0))
        # Avoid division by zero; if any norm is 0 (unlikely), leave vector unchanged
        nz = norms > 0
        Xvecs[:, nz] /= norms[nz]

        # Convert to required list formats
        eigenvalues_list: List[float] = eigvals.tolist()
        eigenvectors_list: List[List[float]] = [Xvecs[:, i].tolist() for i in range(Xvecs.shape[1])]

        return (eigenvalues_list, eigenvectors_list)