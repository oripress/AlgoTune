import os

# Set BLAS / threading env vars early to avoid oversubscription and reduce thread startup overhead.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_MAIN_FREE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        Optimized matrix multiplication.

        - Sets BLAS/thread env vars early (module import time).
        - Minimizes unnecessary copies by checking input types and memory layout.
        - Preallocates the output and uses np.dot(..., out=...) to leverage BLAS efficiently.
        - Returns a list of 1D numpy arrays (rows) which is cheap to produce and compatible.
        """
        if not isinstance(problem, dict):
            raise ValueError("problem must be a dict with keys 'A' and 'B'")
        if "A" not in problem or "B" not in problem:
            raise ValueError("problem must contain 'A' and 'B'")

        A_in = problem["A"]
        B_in = problem["B"]

        # Efficiently obtain float64, C-contiguous arrays without copying when possible.
        if isinstance(A_in, np.ndarray):
            A = A_in
            if A.dtype != np.float64:
                A = A.astype(np.float64, copy=False)
            if A.ndim != 2:
                A = np.atleast_2d(A)
            if not A.flags["C_CONTIGUOUS"]:
                A = np.ascontiguousarray(A)
        else:
            A = np.array(A_in, dtype=np.float64, order="C")

        if isinstance(B_in, np.ndarray):
            B = B_in
            if B.dtype != np.float64:
                B = B.astype(np.float64, copy=False)
            if B.ndim != 2:
                B = np.atleast_2d(B)
            if not B.flags["C_CONTIGUOUS"]:
                B = np.ascontiguousarray(B)
        else:
            B = np.array(B_in, dtype=np.float64, order="C")

        # Validate shapes
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Incompatible shapes for multiplication: {A.shape} and {B.shape}")

        n, p = A.shape[0], B.shape[1]

        # Preallocate output and perform BLAS-backed multiplication into it.
        C = np.empty((n, p), dtype=np.float64, order="C")
        np.dot(A, B, out=C)

        # Return a list of row arrays (cheap — no deep copy of elements).
        return list(C)