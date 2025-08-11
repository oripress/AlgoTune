import numpy as np
from typing import Any, Dict
from scipy.linalg import expm

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Compute the communicability matrix exp(A) for an undirected graph given by its adjacency list.
        Returns a nested dictionary {u: {v: value}} with float entries.
        """
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)

        # Empty graph case
        if n == 0:
            return {"communicability": {}}

        # Build dense adjacency matrix (ensure symmetry)
        A = np.zeros((n, n), dtype=np.float64)
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                A[u, v] = 1.0
                A[v, u] = 1.0  # ensure symmetry

        # Compute matrix exponential using dense routine (matches reference)
        expA = expm(A)

        # Build nested dict output
        comm_dict: Dict[int, Dict[int, float]] = {}
        for i in range(n):
            row = expA[i]
            inner: Dict[int, float] = {}
            for j in range(n):
                inner[j] = float(row[j])
            comm_dict[i] = inner

        return {"communicability": comm_dict}