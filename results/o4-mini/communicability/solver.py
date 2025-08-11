import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)
        if n == 0:
            return {"communicability": {}}
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=float)
        for i, nbrs in enumerate(adj_list):
            if nbrs:
                A[i, nbrs] = 1.0
        # Eigen-decomposition for symmetric A
        w, V = np.linalg.eigh(A)
        exp_w = np.exp(w)
        # Compute exp(A) = V * diag(exp_w) * V^T
        expA = (V * exp_w) @ V.T
        # Convert to nested dict
        rows = expA.tolist()
        comm = {u: dict(enumerate(rows[u])) for u in range(n)}
        return {"communicability": comm}