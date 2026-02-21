import numpy as np
from fast_qz import compute_qz

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        AA, BB, Q, Z = compute_qz(A, B)
        return {
            "QZ": {
                "AA": AA,
                "BB": BB,
                "Q": Q,
                "Z": Z
            }
        }