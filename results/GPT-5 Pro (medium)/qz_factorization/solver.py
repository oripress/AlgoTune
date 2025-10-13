from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import qz

class Solver:
    def solve(self, problem: dict[str, list[list[float]]], **kwargs: Any) -> dict[str, Any]:
        """
        Compute the QZ (generalized Schur) factorization of matrices A and B.

        Given A and B (n x n), find unitary Q and Z such that:
            A = Q @ AA @ Z.conj().T
            B = Q @ BB @ Z.conj().T
        where AA and BB are quasi-upper-triangular for real input (AA with 1x1/2x2 blocks)
        and upper triangular for complex input.

        Returns:
            A dictionary with key "QZ" containing "AA", "BB", "Q", and "Z" as lists.
        """
        A_np = np.asarray(problem["A"])
        B_np = np.asarray(problem["B"])

        # Determine if complex output is needed
        is_complex = np.iscomplexobj(A_np) or np.iscomplexobj(B_np)
        dtype = np.complex128 if is_complex else np.float64

        # Ensure Fortran-contiguous arrays for LAPACK efficiency
        A = np.asfortranarray(A_np, dtype=dtype)
        B = np.asfortranarray(B_np, dtype=dtype)

        # Compute QZ factorization
        # Use check_finite=False and overwrite for performance.
        out_mode = "complex" if is_complex else "real"
        AA, BB, Q, Z = qz(
            A,
            B,
            output=out_mode,
            overwrite_a=True,
            overwrite_b=True,
            check_finite=False,
        )

        return {
            "QZ": {
                "AA": AA.tolist(),
                "BB": BB.tolist(),
                "Q": Q.tolist(),
                "Z": Z.tolist(),
            }
        }