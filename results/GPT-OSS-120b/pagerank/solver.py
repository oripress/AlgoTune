from __future__ import annotations
from typing import Any, Dict, List

import numpy as np

class Solver:
    """
    Fast PageRank implementation using pure NumPy.

    The algorithm follows the classic power‑iteration method:
        r_{k+1} = α·P·r_k + (1−α)·(1/n)·1

    where P is the column‑stochastic transition matrix implied by the
    adjacency list.  The implementation avoids constructing P explicitly.
    It handles dangling nodes by treating them as linking uniformly to all
    nodes.

    Parameters
    ----------
    alpha : float, optional (default=0.85)
        Damping factor.
    tol : float, optional (default=1e-8)
        Convergence tolerance measured in L1 norm.
    max_iter : int, optional (default=100)
        Maximum number of power‑iteration steps.
    """

    alpha: float = 0.85
    tol: float = 1e-8
    max_iter: int = 100

    def solve(self, problem: Dict[str, List[List[int]]], **kwargs) -> Dict[str, List[float]]:
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)

        # Edge‑case handling
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        # Pre‑compute out‑degrees and identify dangling nodes
        out_deg = np.empty(n, dtype=np.int64)
        dangling = np.empty(n, dtype=bool)
        for i, neighbors in enumerate(adj_list):
            d = len(neighbors)
            out_deg[i] = d
            dangling[i] = (d == 0)

        # Initialise rank vector with uniform distribution
        r = np.full(n, 1.0 / n, dtype=np.float64)

        # Pre‑compute constant teleport term
        teleport = (1.0 - self.alpha) / n

        for _ in range(self.max_iter):
            # Contribution from dangling nodes (they link uniformly to all nodes)
            dangling_sum = r[dangling].sum()
            # Start with teleport + dangling contribution (both uniform)
            new_r = np.full(n, teleport + self.alpha * dangling_sum / n, dtype=np.float64)

            # Add contributions from regular outgoing links
            for i, neighbors in enumerate(adj_list):
                d = out_deg[i]
                if d == 0:
                    continue
                contrib = r[i] / d
                new_r[neighbors] += self.alpha * contrib

            # Convergence test (L1 norm)
            if np.abs(new_r - r).sum() < self.tol:
                r = new_r
                break
            r = new_r

        # Normalise to ensure sum == 1 (tiny drift possible)
        r /= r.sum()

        return {"pagerank_scores": r.tolist()}