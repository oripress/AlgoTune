from __future__ import annotations

from typing import Any, Dict

import numpy as np

class Solver:
    def solve(self, problem: Dict[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Compute the projection of a symmetric matrix A onto the PSD cone.

        Args:
            problem: A dictionary with key "A" representing an n x n symmetric matrix.

        Returns:
            A dictionary with key "X": the PSD projection of A.
        """
        A = np.asarray(problem["A"])
        # Use eigh for symmetric matrices (faster and numerically stable)
        # Assume A is symmetric as per problem statement.
        # If A might be slightly non-symmetric due to numeric noise, eigh
        # effectively uses one triangle, which is fine for this task.
        w, V = np.linalg.eigh(A)

        # Clip eigenvalues at zero
        w = np.maximum(w, 0.0)

        # If there are no positive eigenvalues, return zero matrix quickly
        if np.all(w == 0):
            X = np.zeros_like(A, dtype=A.dtype)
            return {"X": X}

        # Use only the columns corresponding to positive eigenvalues (saves work if many negatives)
        pos = w > 0
        if np.any(~pos):
            Vp = V[:, pos]
            wp = w[pos]
            # Reconstruct X = Vp diag(wp) Vp^T efficiently
            X = (Vp * wp) @ Vp.T
        else:
            # All eigenvalues nonnegative
            X = (V * w) @ V.T

        return {"X": X}