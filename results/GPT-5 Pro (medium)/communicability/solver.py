from typing import Any, Dict, List

import numpy as np


class Solver:
    def solve(self, problem: Dict[str, List[List[int]]], **kwargs) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Calculate communicability C(u, v) = (e^A)_{uv} for an undirected graph
        given by its adjacency list.

        Uses symmetric eigendecomposition for fast matrix exponential:
            For symmetric A: A = V diag(w) V^T  =>  e^A = V diag(exp(w)) V^T
        """
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)

        # Handle empty graph
        if n == 0:
            return {"communicability": {}}

        # Build symmetric adjacency matrix A (float64)
        A = np.zeros((n, n), dtype=np.float64)
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                # Add each undirected edge once to avoid redundant writes
                if u < v:
                    A[u, v] = 1.0
                    A[v, u] = 1.0

        # Compute e^A via symmetric eigendecomposition
        # A is symmetric for undirected graphs
        w, V = np.linalg.eigh(A)  # A = V diag(w) V^T
        e_w = np.exp(w)
        V_scaled = V * e_w  # scale columns of V by exp(eigenvalues)
        expA = V_scaled @ V.T  # e^A

        # Convert to the required dict-of-dicts format with standard Python floats
        comm: Dict[int, Dict[int, float]] = {}
        expA_list = expA.tolist()  # nested list of Python floats
        for i in range(n):
            row = expA_list[i]
            comm[i] = {j: float(row[j]) for j in range(n)}

        return {"communicability": comm}