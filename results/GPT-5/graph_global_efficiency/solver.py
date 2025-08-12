from typing import Any, List

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

if njit is not None:
    @njit(cache=True)
    def _sum_inverse_ordered(n, offsets, neighbors, recips, sources):
        # BFS from selected sources using queue arrays and visitation marks.
        seen = np.zeros(n, dtype=np.int64)
        q = np.empty(n, dtype=np.int64)
        total = 0.0
        mark = np.int64(0)

        ns = sources.shape[0]
        for si in range(ns):
            s = sources[si]
            mark += 1
            seen[s] = mark
            q_head = np.int64(0)
            q_tail = np.int64(1)
            q[0] = s
            dist = np.int64(0)

            while q_head < q_tail:
                last = q_tail
                dist += 1

                for i in range(q_head, last):
                    u = q[i]
                    start = offsets[u]
                    end = offsets[u + 1]
                    for idx in range(start, end):
                        v = neighbors[idx]
                        if seen[v] != mark:
                            seen[v] = mark
                            q[q_tail] = v
                            q_tail += 1

                total += recips[dist] * (q_tail - last)
                q_head = last

        return total
else:
    def _sum_inverse_ordered(n, offsets, neighbors, recips, sources):
        # Python fallback (layered BFS with CSR traversal)
        seen = [0] * n
        total = 0.0
        mark = 0
        for s in sources:
            mark += 1
            seen[s] = mark
            frontier = [s]
            dist = 1
            while frontier:
                next_frontier: List[int] = []
                for u in frontier:
                    start = int(offsets[u])
                    end = int(offsets[u + 1])
                    for idx in range(start, end):
                        v = int(neighbors[idx])
                        if seen[v] != mark:
                            seen[v] = mark
                            next_frontier.append(v)
                total += recips[dist] * len(next_frontier)
                frontier = next_frontier
                dist += 1
        return total

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        adj: List[List[int]] = problem["adjacency_list"]
        n = len(adj)
        if n <= 1:
            return {"global_efficiency": 0.0}

        # Degrees and quick exits
        deg_list = [len(row) for row in adj]
        total_neighbors = 0
        complete = True
        any_edges = False
        for d in deg_list:
            total_neighbors += d
            if d != n - 1:
                complete = False
            if d > 0:
                any_edges = True

        if not any_edges:
            return {"global_efficiency": 0.0}
        if complete:
            return {"global_efficiency": 1.0}

        # Build CSR offsets and neighbors (int64 for performance on 64-bit)
        offsets = np.empty(n + 1, dtype=np.int64)
        neighbors = np.empty(total_neighbors, dtype=np.int64)
        pos = 0
        offsets[0] = 0
        for i, row in enumerate(adj):
            ln = len(row)
            if ln:
                neighbors[pos : pos + ln] = row
                pos += ln
            offsets[i + 1] = pos

        # Precompute reciprocals: recips[k] = 1/k
        recips = np.empty(n + 1, dtype=np.float64)
        recips[0] = 0.0
        if n >= 1:
            recips[1:] = 1.0 / np.arange(1, n + 1, dtype=np.float64)

        # Sources: skip isolated vertices (deg == 0)
        nz = sum(1 for d in deg_list if d > 0)
        sources = np.fromiter((i for i, d in enumerate(deg_list) if d > 0), dtype=np.int64, count=nz)

        total = _sum_inverse_ordered(np.int64(n), offsets, neighbors, recips, sources)
        denom = float(n * (n - 1))
        return {"global_efficiency": float(total / denom)}