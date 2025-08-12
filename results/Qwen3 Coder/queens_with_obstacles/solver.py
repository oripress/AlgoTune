import numpy as np
from typing import List, Tuple

class Solver:
    def solve(self, problem: np.ndarray, **kwargs) -> List[Tuple[int, int]]:
        """
        Solves the Queens with Obstacles Problem using a greedy approach.
        """
        if problem is None or problem.size == 0:
            return []
        
        n, m = problem.shape
        if n == 0 or m == 0:
            return []
        
        # Simple greedy approach - place queens one by one
        solution = []
        
        # Try all positions in order
        for r in range(n):
            for c in range(m):
                if not problem[r, c]:  # If not an obstacle
                    # Check if this position conflicts with any queen already placed
                    conflict = False
                    for qr, qc in solution:
                        # Same row, column, or diagonal
                        if r == qr or c == qc or abs(r - qr) == abs(c - qc):
                            conflict = True
                            break
                    if not conflict:
                        solution.append((r, c))
        
        return solution