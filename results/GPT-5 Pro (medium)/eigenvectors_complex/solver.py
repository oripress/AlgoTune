from __future__ import annotations

from typing import Any, List

import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> List[List[complex]]:
        """
        Compute eigenvectors of a real (possibly non-symmetric) square matrix.

        Returns:
            A list of normalized eigenvectors (each a list of complex numbers),
            sorted to match the reference:
              - Sort eigenpairs in descending order by eigenvalue real part,
                then by imaginary part.
        """
        A = np.asarray(problem)
        # Compute eigenvalues and right eigenvectors; eigenvectors are columns
        w, v = np.linalg.eig(A)

        # Stable sort to match reference ordering:
        # 1) stable sort by secondary key: -imag (descending)
        idx = np.argsort(-w.imag, kind="mergesort")
        # 2) stable sort by primary key: -real (descending), preserving step 1 for ties
        primary_keys = (-w.real)[idx]
        idx = idx[np.argsort(primary_keys, kind="mergesort")]

        # Reorder eigenvectors by idx
        v = v[:, idx]

        # Normalize each eigenvector to unit Euclidean norm
        norms = np.linalg.norm(v, axis=0)
        # Guard against numerical underflow (though eigenvectors shouldn't be zero)
        small = norms < 1e-12
        if np.any(small):
            norms = norms.copy()
            norms[small] = 1.0
        v = v / norms

        # Return as list of lists (each eigenvector is a column)
        return [v[:, i].tolist() for i in range(v.shape[1])]