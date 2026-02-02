import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        A = problem["matrix"]
        if not isinstance(A, np.ndarray):
            A = np.array(A)
            
        Q, R = np.linalg.qr(A, mode="reduced")
        return {"QR": {"Q": Q, "R": R}}