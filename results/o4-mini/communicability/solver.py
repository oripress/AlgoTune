import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem, **kwargs):
        adj_list = problem.get("adjacency_list") or []
        n = len(adj_list)
        if n == 0:
            return {"communicability": {}}
        # Build adjacency matrix
        counts = [len(neighbors) for neighbors in adj_list]
        total = sum(counts)
        if total > 0:
            # Use vectorized assignment for speed
            rows = np.repeat(np.arange(n, dtype=int), counts)
            cols = np.concatenate(adj_list).astype(int)
            A = np.zeros((n, n), dtype=np.float64)
            A[rows, cols] = 1.0
        else:
            # No edges: zero matrix
            A = np.zeros((n, n), dtype=np.float64)
        # Compute full matrix exponential
        M = expm(A)
        # Convert to nested dict-of-dicts
        rows_list = M.tolist()  # convert to Python floats
        comm = {i: dict(enumerate(rows_list[i])) for i in range(n)}
        return {"communicability": comm}