from typing import Any, Dict
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Project a symmetric matrix A onto the PSD cone:
            X = U * diag(max(eigvals, 0)) * U^T

        Args:
            problem: dict with key "A" containing an n x n symmetric matrix
                     (list of lists or array-like).

        Returns:
            dict with key "X" containing the projected PSD matrix as nested lists.
        """
        if "A" not in problem:
            raise KeyError("Problem must contain key 'A'.")

        A = np.asarray(problem["A"], dtype=float)

        # Handle empty input
        if A.size == 0:
            return {"X": []}

        # Scalars and 1-element arrays -> 1x1 matrix
        if A.ndim == 0:
            v = float(A)
            return {"X": [[v if v > 0.0 else 0.0]]}
        if A.ndim == 1:
            if A.size == 1:
                v = float(A[0])
                return {"X": [[v if v > 0.0 else 0.0]]}
            raise ValueError("Input matrix A must be square (n x n).")

        # Ensure square matrix
        if A.shape[0] != A.shape[1]:
            raise ValueError("Input matrix A must be square (n x n).")

        # Symmetrize to reduce numerical asymmetry
        A = 0.5 * (A + A.T)

        n = A.shape[0]
        if n == 1:
            v = float(A[0, 0])
            return {"X": [[v if v > 0.0 else 0.0]]}

        # Use symmetric eigendecomposition (fast and stable for real symmetric matrices)
        try:
            eigvals, eigvecs = np.linalg.eigh(A)
        except np.linalg.LinAlgError:
            # Fallback to general decomposition; take real parts
            eigvals, eigvecs = np.linalg.eig(A)
            eigvals = np.real(eigvals)
            eigvecs = np.real(eigvecs)

        # Clip negatives to zero
        eigvals_clipped = np.clip(eigvals, 0.0, None)

        # Numerical tolerance to ignore tiny eigenvalues
        max_abs_eig = np.max(np.abs(eigvals)) if eigvals.size else 0.0
        tol = max(1e-12, max_abs_eig * np.finfo(float).eps * n)

        pos_mask = eigvals_clipped > tol

        if not np.any(pos_mask):
            X = np.zeros((n, n), dtype=float)
        else:
            V = eigvecs[:, pos_mask]       # n x k
            L = eigvals_clipped[pos_mask]  # k
            # Efficient reconstruction: X = V * diag(L) * V^T
            X = (V * L) @ V.T

        # Ensure symmetry and real values (cleanup numerical noise)
        X = np.real(0.5 * (X + X.T))

        return {"X": X.tolist()}