import math
import heapq
from typing import Any, Dict, List, Tuple

import numpy as np

# Optional SciPy for a fast C implementation; fall back to Python if unavailable.
try:
    import scipy.sparse
    import scipy.sparse.csgraph
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False

class Solver:
    def __init__(self):
        # No heavy initialization required.
        pass

    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        Compute all-pairs shortest paths for a CSR graph.
        Input problem dict must contain: data, indices, indptr, shape.
        Returns {"distance_matrix": list[list[float|None]]} with None representing unreachable pairs.
        Optional kwarg: directed (bool). If absent, defaults to False (undirected).
        """
        # Basic validation
        if not isinstance(problem, dict):
            return {"distance_matrix": []}
        if not all(k in problem for k in ("data", "indices", "indptr", "shape")):
            return {"distance_matrix": []}

        try:
            shape = problem["shape"]
            n = int(shape[0])
        except Exception:
            return {"distance_matrix": []}

        if n == 0:
            return {"distance_matrix": []}
        if n == 1:
            # Single node: trivial
            return {"distance_matrix": [[0.0]]}

        try:
            data = np.array(problem["data"], dtype=np.float64)
            indices = np.array(problem["indices"], dtype=np.int64)
            indptr = np.array(problem["indptr"], dtype=np.int64)
        except Exception:
            return {"distance_matrix": []}

        directed = bool(kwargs.get("directed", problem.get("directed", False)))

        # Validate CSR pointers
        if indptr.shape[0] != n + 1:
            return {"distance_matrix": []}

        # No edges case
        if data.size == 0:
            out = [[0.0 if i == j else None for j in range(n)] for i in range(n)]
            return {"distance_matrix": out}

        # Try SciPy implementation for speed: use specialized dijkstra call and vectorized postprocessing
        if _SCIPY_AVAILABLE:
            try:
                csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=(n, n))
                # Use Dijkstra (C implementation) for sparse graphs
                dist_matrix = scipy.sparse.csgraph.dijkstra(csgraph=csr, directed=directed, indices=None)
                # Ensure 2D numpy float array
                dist_matrix = np.atleast_2d(np.asarray(dist_matrix, dtype=np.float64))
                try:
                    # Fast path: convert to object array and set infinities to None
                    obj = dist_matrix.astype(object)
                    mask = ~np.isfinite(dist_matrix)
                    if mask.any():
                        obj[mask] = None
                    result = obj.tolist()
                    return {"distance_matrix": result}
                except Exception:
                    # Fallback: row-wise safe conversion
                    result: List[List[Any]] = []
                    for i in range(n):
                        row = dist_matrix[i]
                        if np.isfinite(row).all():
                            result.append([float(x) for x in row.tolist()])
                        else:
                            result.append([(None if not np.isfinite(x) else float(x)) for x in row.tolist()])
                    return {"distance_matrix": result}
            except Exception:
                # Fall through to Python fallback
                pass
                # Fall through to Python fallback
                pass

        # Build adjacency list from CSR for Python fallback
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for u in range(n):
            start = int(indptr[u])
            end = int(indptr[u + 1])
            for k in range(start, end):
                v = int(indices[k])
                w = float(data[k])
                adj[u].append((v, w))
                if not directed:
                    adj[v].append((u, w))

        # Dijkstra per source (Python fallback)
        result: List[List[Any]] = []
        for s in range(n):
            dist = [math.inf] * n
            dist[s] = 0.0
            visited = [False] * n
            heap: List[Tuple[float, int]] = [(0.0, s)]
            while heap:
                d_u, u = heapq.heappop(heap)
                if visited[u]:
                    continue
                visited[u] = True
                if d_u > dist[u]:
                    continue
                for v, w in adj[u]:
                    nd = d_u + w
                    if nd < dist[v]:
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))
            row = [None if math.isinf(dist[j]) else float(dist[j]) for j in range(n)]
            result.append(row)

        return {"distance_matrix": result}