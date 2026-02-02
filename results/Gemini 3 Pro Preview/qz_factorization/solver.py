import numpy as np
from scipy.linalg import qz
import sys

try:
    import fast_solver
except ImportError as e:
    print(f"Failed to import fast_solver: {e}", file=sys.stderr)
    fast_solver = None

class Solver:
    def solve(self, problem: dict[str, list[list[float]]], **kwargs) -> dict[str, dict[str, list[list[float | complex]]]]:
        if fast_solver is not None:
            # fast_solver.solve_qz takes lists and returns lists
            AA, BB, Q, Z = fast_solver.solve_qz(problem["A"], problem["B"])
            solution = {"QZ": {"AA": AA, "BB": BB, "Q": Q, "Z": Z}}
        else:
            # Fallback to scipy.linalg.qz
            A = np.array(problem["A"])
            B = np.array(problem["B"])
            AA, BB, Q, Z = qz(A, B, output="real")
            solution = {"QZ": {"AA": AA.tolist(), "BB": BB.tolist(), "Q": Q.tolist(), "Z": Z.tolist()}}
        
        return solution