from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

class Solver:
    def __init__(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> None:
        # Match NetworkX defaults
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def solve(self, problem: Dict[str, List[List[int]]], **kwargs) -> Dict[str, List[float]]:
        """
        Compute PageRank using an efficient power-iteration with vectorized operations.
        Mirrors NetworkX default parameters and behavior for uniform personalization and
        uniform dangling distribution.

        Args:
            problem: {"adjacency_list": List[List[int]]}
            kwargs: optional overrides for alpha, max_iter, tol

        Returns:
            {"pagerank_scores": List[float]}
        """
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)

        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            # Single node graph has rank 1.0
            return {"pagerank_scores": [1.0]}

        # Allow optional overrides
        alpha = float(kwargs.get("alpha", self.alpha))
        max_iter = int(kwargs.get("max_iter", self.max_iter))
        tol = float(kwargs.get("tol", self.tol))

        # First pass: compute out-degrees (deduplicated neighbors to mirror NetworkX DiGraph behavior)
        degrees = np.empty(n, dtype=np.int64)
        m = 0
        dedup_lists: List[List[int]] = [None] * n  # type: ignore[assignment]
        for u, neigh in enumerate(adj_list):
            if not neigh:
                degrees[u] = 0
                dedup_lists[u] = []
                continue
            # adjacency_list[i] is sorted -> deduplicate with linear scan
            uniq = []
            last = None
            for v in neigh:
                if v != last:
                    uniq.append(v)
                    last = v
            d = len(uniq)
            degrees[u] = d
            dedup_lists[u] = uniq
            m += d

        # Build edge arrays if any edges exist
        if m > 0:
            src = np.empty(m, dtype=np.int64)
            dst = np.empty(m, dtype=np.int64)
            pos = 0
            for u in range(n):
                neigh = dedup_lists[u]
                d = degrees[u]
                if d:
                    end = pos + d
                    src[pos:end] = u
                    # Copy neighbors into dst
                    dst[pos:end] = np.fromiter(neigh, dtype=np.int64, count=d)
                    pos = end
        else:
            # Dummy arrays; not used when m == 0
            src = np.empty(0, dtype=np.int64)
            dst = np.empty(0, dtype=np.int64)

        # Initial rank vector: uniform distribution
        x = np.full(n, 1.0 / n, dtype=np.float64)

        # Precompute mask for dangling nodes
        dangling_mask = degrees == 0

        # Iterative power method
        # Convergence criterion matches NetworkX: sum(|x - xlast|) < n * tol
        teleport = (1.0 - alpha) / n

        for _ in range(max_iter):
            xlast = x

            # Contribution from links
            if m > 0:
                deg_src = degrees[src].astype(np.float64, copy=False)  # deg_src > 0
                link_weights = xlast[src] / deg_src
                link_sum = np.bincount(dst, weights=link_weights, minlength=n)
            else:
                link_sum = np.zeros(n, dtype=np.float64)

            # Contribution from dangling nodes (uniform)
            dangling_weight = float(xlast[dangling_mask].sum())

            base = teleport + alpha * dangling_weight / n

            x = base + alpha * link_sum

            # Check convergence
            err = np.abs(x - xlast).sum()
            if err < n * tol:
                break

        # Return as list of floats
        return {"pagerank_scores": x.tolist()}