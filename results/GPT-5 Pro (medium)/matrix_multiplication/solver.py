from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[List[float]]:
        """
        Compute matrix product C = A · B.

        Input:
            problem: dict with keys "A" and "B" as list-of-lists or NumPy arrays.

        Output:
            A list of lists representing the resulting matrix C.
        """
        A = problem["A"]
        B = problem["B"]

        # Convert inputs to numpy arrays without unnecessary copying
        A_np = np.asarray(A)
        B_np = np.asarray(B)

        # Use NumPy's highly optimized matrix multiplication
        C = np.dot(A_np, B_np)

        # Return as list of lists to satisfy validator type check
        return C.tolist()