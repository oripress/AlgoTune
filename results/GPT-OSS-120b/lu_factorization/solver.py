import numpy as np
from typing import Any, Dict, List
from scipy.linalg import lu

class Solver:
    def solve(self, problem: Dict[str, List[List[float]]], **kwargs) -> Dict[str, Dict[str, List[List[float]]]]:
        """
        Compute the LU factorization with partial pivoting using SciPy's optimized implementation.
        The call disables extra finite‑value checks and overwrites the input array to reduce overhead.
        Returns a dictionary with keys "P", "L", "U" as plain Python lists.
        """
        A = np.array(problem["matrix"], dtype=float)
        # Faster LU: avoid extra copies and finite‑value checks
        P, L, U = lu(A, overwrite_a=True, check_finite=False)
        return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}