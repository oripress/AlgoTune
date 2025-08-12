from typing import Any, List
import numpy as np

class Solver:
    def __init__(self):
        # Prepare optional SciPy fast path (done here so import cost is not charged to solve).
        self._shortest_path = None
        self._csr_matrix = None
        try:
            from scipy.sparse.csgraph import shortest_path  # type: ignore
            from scipy.sparse import csr_matrix  # type: ignore
            self._shortest_path = shortest_path
            self._csr_matrix = csr_matrix
        except Exception:
            self._shortest_path = None
            self._csr_matrix = None

    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Compute global efficiency for an undirected, unweighted graph given as adjacency_list.
        Returns {"global_efficiency": float}.
        """
        adj = problem.get("adjacency_list", [])
        n = len(adj)
        if n <= 1:
            return {"global_efficiency": 0.0}

        degrees = [len(nei) for nei in adj]
        m = sum(degrees)

        # Quick trivial checks
        if m == 0:
            return {"global_efficiency": 0.0}
        if all(d == n - 1 for d in degrees):
            return {"global_efficiency": 1.0}

        # Use SciPy's C implementation for moderate-sized problems (fast).
        # Threshold chosen to avoid excessive memory usage for very large n.
        if self._shortest_path is not None and n <= 3000:
            # Build COO-like arrays efficiently into numpy arrays
            rows = np.empty(m, dtype=np.int32)
            cols = np.empty(m, dtype=np.int32)
            p = 0
            for i, nbrs in enumerate(adj):
                l = len(nbrs)
                if l:
                    rows[p : p + l] = i
                    cols[p : p + l] = nbrs  # assign list slice-to-numpy
                    p += l
            data = np.ones(m, dtype=np.int8)
            csr = self._csr_matrix((data, (rows, cols)), shape=(n, n))
            # Compute all-pairs shortest paths (unweighted -> BFS internally)
            dist = self._shortest_path(csr, directed=False, unweighted=True)
            # Sum inverse distances for finite, non-zero distances
            with np.errstate(divide="ignore", invalid="ignore"):
                mask = (dist > 0) & np.isfinite(dist)
                inv = np.zeros_like(dist, dtype=np.float64)
                inv[mask] = 1.0 / dist[mask]
                total_inv = float(inv.sum())
            efficiency = total_inv / (n * (n - 1))
            return {"global_efficiency": float(efficiency)}

        # Fallback: optimized pure-Python BFS over adjacency lists.
        # Timestamped 'seen' avoids reinitializing arrays each BFS.
        total_inv = 0.0
        seen = [0] * n
        dist = [0] * n
        bfs_id = 1
        adj_local = adj  # local ref for speed

        for src in range(n):
            seen[src] = bfs_id
            dist[src] = 0
            q = [src]
            qi = 0
            append_q = q.append
            seen_local = seen
            dist_local = dist
            while qi < len(q):
                v = q[qi]
                qi += 1
                dv = dist_local[v]
                nd = dv + 1
                inv_nd = 1.0 / nd
                for w in adj_local[v]:
                    if seen_local[w] != bfs_id:
                        seen_local[w] = bfs_id
                        dist_local[w] = nd
                        append_q(w)
                        total_inv += inv_nd
            bfs_id += 1
            # Extremely unlikely overflow guard for bfs_id
            if bfs_id == (1 << 60):
                seen = [0] * n
                dist = [0] * n
                bfs_id = 1

        efficiency = total_inv / (n * (n - 1))
        return {"global_efficiency": float(efficiency)}