from __future__ import annotations

from itertools import chain
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import numba as nb
except Exception:  # pragma: no cover
    nb = None  # type: ignore

def _build_csr_from_adjlist(adj: List[List[int]]) -> tuple[np.ndarray, np.ndarray]:
    n = len(adj)
    # Build indptr via cumsum of degrees (fast in NumPy).
    deg = np.fromiter((len(nei) for nei in adj), dtype=np.int32, count=n)
    indptr = np.empty(n + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(deg, out=indptr[1:])
    m = int(indptr[-1])
    if m == 0:
        return indptr, np.empty(0, dtype=np.int32)

    # Flatten all neighbors in one pass.
    indices = np.fromiter(chain.from_iterable(adj), dtype=np.int32, count=m)
    return indptr, indices

def _ensure_inv_np(n: int, inv: Optional[np.ndarray]) -> np.ndarray:
    # Need inv up to index n (safe upper bound for distance).
    if inv is not None and inv.shape[0] > n:
        return inv

    if inv is None:
        out = np.empty(n + 1, dtype=np.float64)
        out[0] = 0.0
        out[1:] = 1.0 / np.arange(1, n + 1, dtype=np.float64)
        return out

    old = inv.shape[0] - 1
    out = np.empty(n + 1, dtype=np.float64)
    out[: old + 1] = inv
    if n > old:
        out[old + 1 :] = 1.0 / np.arange(old + 1, n + 1, dtype=np.float64)
    return out
def _make_numba_kernel():
    if nb is None:
        return None

    @nb.njit(cache=True, fastmath=True)
    def _global_eff_csr(indptr: np.ndarray, indices: np.ndarray, inv: np.ndarray) -> float:
        n = indptr.shape[0] - 1
        if n <= 1:
            return 0.0

        vis = np.zeros(n, dtype=np.int32)
        queue = np.empty(n, dtype=np.int32)

        token = 0
        total = 0.0
        for s in range(n):
            # Isolated nodes contribute 0 as a source; skip quickly.
            if indptr[s] == indptr[s + 1]:
                continue

            token += 1
            vis[s] = token

            head = 0
            tail = 1
            level_end = 1
            dist_u = 0
            queue[0] = s

            while head < tail:
                if head == level_end:
                    dist_u += 1
                    level_end = tail
                u = queue[head]
                head += 1

                nd = dist_u + 1
                inv_nd = inv[nd]
                start = indptr[u]
                end = indptr[u + 1]
                for k in range(start, end):
                    v = indices[k]
                    if vis[v] != token:
                        vis[v] = token
                        total += inv_nd
                        queue[tail] = v
                        tail += 1
        return total / (n * (n - 1))

    return _global_eff_csr

class Solver:
    """
    Fast global efficiency for an undirected, unweighted graph given as adjacency lists.

    Uses a Numba-compiled BFS over CSR adjacency for speed.
    """

    __slots__ = ("_inv_np", "_kernel")

    def __init__(self) -> None:
        # Precompile kernel in init (compilation time excluded from solve runtime).
        self._inv_np: Optional[np.ndarray] = None
        self._kernel = _make_numba_kernel()
        if self._kernel is not None:
            # Trigger compilation with a tiny dummy graph.
            indptr = np.array([0, 0], dtype=np.int32)
            indices = np.array([], dtype=np.int32)
            inv = np.array([0.0, 1.0], dtype=np.float64)
            self._kernel(indptr, indices, inv)

    @staticmethod
    def _py_fallback(adj: List[List[int]]) -> float:
        n = len(adj)
        if n <= 1:
            return 0.0

        inv = [0.0] * (n + 1)
        for k in range(1, n + 1):
            inv[k] = 1.0 / k

        vis = [0] * n
        dist = [0] * n
        token = 0
        total = 0.0

        for s in range(n):
            token += 1
            vis[s] = token
            dist[s] = 0
            q = [s]
            head = 0
            append = q.append
            while head < len(q):
                u = q[head]
                head += 1
                du = dist[u] + 1
                for v in adj[u]:
                    if vis[v] != token:
                        vis[v] = token
                        dist[v] = du
                        total += inv[du]
                        append(v)

        return total / (n * (n - 1))

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, float]:
        adj: List[List[int]] = problem["adjacency_list"]
        n = len(adj)
        if n <= 1:
            return {"global_efficiency": 0.0}

        # For tiny graphs, conversion + numba call overhead may not pay off.
        if self._kernel is None or n < 32:
            return {"global_efficiency": float(self._py_fallback(adj))}

        indptr, indices = _build_csr_from_adjlist(adj)
        if indices.size == 0:
            return {"global_efficiency": 0.0}

        self._inv_np = _ensure_inv_np(n, self._inv_np)
        eff = self._kernel(indptr, indices, self._inv_np)
        return {"global_efficiency": float(eff)}