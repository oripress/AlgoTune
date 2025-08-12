from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import qz as _qz

class Solver:
    def solve(
        self, problem: dict[str, list[list[float]]], **kwargs: Any
    ) -> dict[str, dict[str, list[list[float]]]]:
        """
        Compute the QZ (generalized Schur) factorization of (A, B):
            A = Q * AA * Z^*
            B = Q * BB * Z^*
        Returns AA, BB (quasi-upper triangular for real input), and unitary Q, Z.

        Performance choices:
        - Use SciPy's qz with check_finite=False and overwrite flags for lower overhead.
        - Use Fortran-order float64 arrays to minimize internal copies in LAPACK calls.
        - Fast trivial paths for n=0 and n=1.
        """
        A_in = problem["A"]
        B_in = problem["B"]

        n = len(A_in)
        if n == 0:
            return {"QZ": {"AA": [], "BB": [], "Q": [], "Z": []}}
        if n == 1:
            a00 = float(A_in[0][0])
            b00 = float(B_in[0][0])
            return {"QZ": {"AA": [[a00]], "BB": [[b00]], "Q": [[1.0]], "Z": [[1.0]]}}

        # Fortran-contiguous float64 arrays for efficient LAPACK usage
        A = np.array(A_in, dtype=np.float64, order="F", copy=True)
        B = np.array(B_in, dtype=np.float64, order="F", copy=True)

        qz = _qz  # local alias to avoid global attribute lookup
        AA, BB, Q, Z = qz(
            A,
            B,
            output="real",
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