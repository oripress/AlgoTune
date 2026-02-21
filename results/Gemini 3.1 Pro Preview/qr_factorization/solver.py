import numpy as np
import scipy.linalg
from typing import Any

class Solver:
    def solve(self, problem: dict[str, np.ndarray], **kwargs) -> Any:
        A = problem["matrix"]
        # Copy A if we want to overwrite it, but since we don't need A anymore, we can just pass it.
        # However, A might be read-only or we might not want to mutate the input dictionary.
        # Let's just use check_finite=False and mode='economic'
        Q, R = scipy.linalg.qr(A, mode="economic", check_finite=False)
        return {"QR": {"Q": Q, "R": R}}