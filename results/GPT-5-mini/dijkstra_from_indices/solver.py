from typing import Any, Dict, List, Optional
import math
import heapq

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[List[Optional[float]]]]:
        """
        Optimized shortest-path solver for CSR graphs.

        - If multiple sources and min_only is True (or default), performs one
          multi-source Dijkstra (faster) and returns a single row with the
          elementwise minimum distances across sources.
        - Otherwise uses SciPy's dijkstra when available for full per-source
          distances; falls back to pure-Python Dijkstra implementations.
        - Unreachable nodes are represented as None.
        """
        # Extract inputs
        try:
            data = problem["data"]
            indices = problem["indices"]
            indptr = problem["indptr"]
            shape = problem["shape"]
            source_indices = problem["source_indices"]
        except Exception:
            return {"distances": []}

        # Validate shape and node count
        if not isinstance(shape, (list, tuple)) or len(shape) < 1:
            return {"distances": []}
        try:
            n = int(shape[0])
        except Exception:
            return {"distances": []}
        if n < 0:
            return {"distances": []}

        # Handle empty graph
        if n == 0:
            if isinstance(source_indices, (list, tuple)):
                return {"distances": [[] for _ in source_indices]}
            if source_indices is None:
                return {"distances": []}
            return {"distances": [[]]}

        # Normalize source indices
        srcs: List[int]
        if isinstance(source_indices, (int,)):
            srcs = [int(source_indices)]
        elif isinstance(source_indices, (list, tuple)):
            try:
                srcs = [int(x) for x in source_indices]
            except Exception:
                return {"distances": []}
        else:
            # Try numpy types
            try:
                import numpy as _np  # lazy import
                if isinstance(source_indices, _np.ndarray):
                    srcs = [int(x) for x in source_indices.tolist()]
                elif isinstance(source_indices, _np.integer):
                    srcs = [int(source_indices)]
                else:
                    return {"distances": []}
            except Exception:
                return {"distances": []}

        if len(srcs) == 0:
            return {"distances": []}
        for s in srcs:
            if s < 0 or s >= n:
                return {"distances": []}

        # Determine whether to collapse to minima (min_only)
        if "min_only" in kwargs:
            min_only = bool(kwargs["min_only"])
        else:
            min_only = bool(getattr(self, "min_only", True if len(srcs) > 1 else False))

        # If multiple sources and collapsing allowed -> run multi-source Dijkstra (single run)
        if len(srcs) > 1 and min_only:
            try:
                distances = self._multi_source_dijkstra(n, indptr, indices, data, srcs)
            except Exception:
                # Fallback to SciPy multi-source if available
                try:
                    import scipy.sparse as _sp
                    import scipy.sparse.csgraph as _csgraph
                    import numpy as _np
                    graph = _sp.csr_matrix((data, indices, indptr), shape=(n, n))
                    dist = _csgraph.dijkstra(csgraph=graph, directed=False, indices=srcs, min_only=True)
                    distances = [None if math.isinf(x) else float(x) for x in (dist.tolist() if hasattr(dist, "tolist") else list(dist))]
                except Exception:
                    return {"distances": []}
            return {"distances": [distances]}

        # Otherwise compute full per-source distances (use SciPy for speed if available)
        try:
            import scipy.sparse as _sp
            import scipy.sparse.csgraph as _csgraph
            graph = _sp.csr_matrix((data, indices, indptr), shape=(n, n))
            distmat = _csgraph.dijkstra(csgraph=graph, directed=False, indices=srcs, min_only=False)
            # Normalize to list-of-lists with None for infinities
            out: List[List[Optional[float]]] = []
            try:
                import numpy as _np
                arr = _np.asarray(distmat)
                if arr.ndim == 1:
                    out.append([None if math.isinf(x) else float(x) for x in arr.tolist()])
                else:
                    for row in arr:
                        out.append([None if math.isinf(x) else float(x) for x in row.tolist()])
            except Exception:
                # Generic iterable handling
                if hasattr(distmat, "__iter__") and not isinstance(distmat, (float, int)):
                    for row in distmat:
                        try:
                            out.append([None if math.isinf(x) else float(x) for x in (row.tolist() if hasattr(row, "tolist") else list(row))])
                        except Exception:
                            return {"distances": []}
                else:
                    # Scalar
                    val = float(distmat)
                    out.append([None if math.isinf(val) else val])
            return {"distances": out}
        except Exception:
            # Fallback: pure-Python Dijkstra per source
            out: List[List[Optional[float]]] = []
            for s in srcs:
                try:
                    out.append(self._single_source_dijkstra(n, indptr, indices, data, s))
                except Exception:
                    return {"distances": []}
            return {"distances": out}

    def _multi_source_dijkstra(self, n, indptr, indices, data, sources):
        """
        Multi-source Dijkstra: initializes heap with all sources (distance 0)
        and computes the minimum distance from any source to every node.
        Returns a Python list of floats or None for unreachable nodes.
        """
        # Convert numpy arrays to lists for faster Python loops if present
        try:
            import numpy as _np
            if isinstance(indptr, _np.ndarray):
                indptr = indptr.tolist()
            if isinstance(indices, _np.ndarray):
                indices = indices.tolist()
            if isinstance(data, _np.ndarray):
                data = data.tolist()
        except Exception:
            pass

        inf = math.inf
        dist = [inf] * n
        heap = []
        heappush = heapq.heappush
        heappop = heapq.heappop

        # Seed heap with unique sources
        for s in set(int(x) for x in sources):
            if dist[s] > 0:
                dist[s] = 0.0
                heappush(heap, (0.0, s))

        indptr_local = indptr
        indices_local = indices
        data_local = data

        while heap:
            d, u = heappop(heap)
            if d != dist[u]:
                continue
            start = indptr_local[u]
            end = indptr_local[u + 1]
            for ei in range(start, end):
                v = indices_local[ei]
                nd = d + float(data_local[ei])
                if nd < dist[v]:
                    dist[v] = nd
                    heappush(heap, (nd, v))

        return [None if math.isinf(x) else float(x) for x in dist]

    def _single_source_dijkstra(self, n, indptr, indices, data, source):
        """
        Standard single-source Dijkstra implementation in pure Python.
        Returns a Python list of floats or None for unreachable nodes.
        """
        try:
            import numpy as _np
            if isinstance(indptr, _np.ndarray):
                indptr = indptr.tolist()
            if isinstance(indices, _np.ndarray):
                indices = indices.tolist()
            if isinstance(data, _np.ndarray):
                data = data.tolist()
        except Exception:
            pass

        inf = math.inf
        dist = [inf] * n
        heap = []
        heappush = heapq.heappush
        heappop = heapq.heappop

        dist[source] = 0.0
        heappush(heap, (0.0, source))

        indptr_local = indptr
        indices_local = indices
        data_local = data

        while heap:
            d, u = heappop(heap)
            if d != dist[u]:
                continue
            start = indptr_local[u]
            end = indptr_local[u + 1]
            for ei in range(start, end):
                v = indices_local[ei]
                nd = d + float(data_local[ei])
                if nd < dist[v]:
                    dist[v] = nd
                    heappush(heap, (nd, v))

        return [None if math.isinf(x) else float(x) for x in dist]