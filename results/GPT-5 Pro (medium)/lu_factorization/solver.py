from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy.linalg import lu

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute LU factorization A = P L U for a given square matrix A.

        Uses scipy.linalg.lu to obtain P, L, U where A = P @ L @ U.

        Returns:
            {
                "LU": {
                    "P": [[...]],
                    "L": [[...]],
                    "U": [[...]],
                }
            }
        """
        A_in = problem["matrix"]
        A = np.asarray(A_in, dtype=float)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Input matrix A must be a square 2D array.")

        P, L, U = lu(A, permute_l=False, overwrite_a=False, check_finite=False)

        solution: Dict[str, Dict[str, List[List[float]]]] = {
            "LU": {
                "P": P.tolist(),
                "L": L.tolist(),
                "U": U.tolist(),
            }
        }
        return solution