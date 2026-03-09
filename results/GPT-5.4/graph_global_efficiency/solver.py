from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import get_num_threads, njit, prange
except Exception:  # pragma: no cover
    get_num_threads = None
    njit = None
    prange = range

if njit is not None:

    @njit(cache=True)
    def _global_efficiency_csr_seq(
        indptr: np.ndarray, indices: np.ndarray, recip: np.ndarray
    ) -> float:
        n = indptr.shape[0] - 1
        if n <= 1:
            return 0.0

        seen = np.zeros(n, dtype=np.int32)
        queue = np.empty(n, dtype=np.int32)

        total = 0.0
        for s in range(n):
            marker = s + 1
            seen[s] = marker
            queue[0] = s
            head = 0
            tail = 1
            depth = 1

            while head < tail:
                level_end = tail
                contrib = recip[depth]

                while head < level_end:
                    u = queue[head]
                    head += 1

                    start = indptr[u]
                    end = indptr[u + 1]
                    for idx in range(start, end):
                        v = indices[idx]
                        if seen[v] != marker:
                            seen[v] = marker
                            total += contrib
                            queue[tail] = v
                            tail += 1

                depth += 1

        return total / (n * (n - 1))

    @njit(cache=True, parallel=True)
    def _global_efficiency_csr_par(
        indptr: np.ndarray, indices: np.ndarray, recip: np.ndarray
    ) -> float:
        n = indptr.shape[0] - 1
        if n <= 1:
            return 0.0

        total = 0.0
        num_threads = get_num_threads()

        for t in prange(num_threads):
            seen = np.zeros(n, dtype=np.int32)
            queue = np.empty(n, dtype=np.int32)
            local = 0.0

            for s in range(t, n, num_threads):
                marker = s + 1
                seen[s] = marker
                queue[0] = s
                head = 0
                tail = 1
                depth = 1

                while head < tail:
                    level_end = tail
                    contrib = recip[depth]

                    while head < level_end:
                        u = queue[head]
                        head += 1

                        start = indptr[u]
                        end = indptr[u + 1]
                        for idx in range(start, end):
                            v = indices[idx]
                            if seen[v] != marker:
                                seen[v] = marker
                                local += contrib
                                queue[tail] = v
                                tail += 1

                    depth += 1

            total += local

        return total / (n * (n - 1))

else:

    def _global_efficiency_csr_seq(
        indptr: np.ndarray, indices: np.ndarray, recip: np.ndarray
    ) -> float:
        n = indptr.shape[0] - 1
        if n <= 1:
            return 0.0

        seen = [0] * n
        queue = [0] * n
        total = 0.0

        for s in range(n):
            marker = s + 1
            seen[s] = marker
            queue[0] = s
            head = 0
            tail = 1
            depth = 1

            while head < tail:
                level_end = tail
                contrib = float(recip[depth])

                while head < level_end:
                    u = queue[head]
                    head += 1

                    start = int(indptr[u])
                    end = int(indptr[u + 1])
                    for idx in range(start, end):
                        v = int(indices[idx])
                        if seen[v] != marker:
                            seen[v] = marker
                            total += contrib
                            queue[tail] = v
                            tail += 1

                depth += 1

        return total / (n * (n - 1))

    def _global_efficiency_csr_par(
        indptr: np.ndarray, indices: np.ndarray, recip: np.ndarray
    ) -> float:
        return _global_efficiency_csr_seq(indptr, indices, recip)

def _build_csr_from_adjacency(
    adjacency_list: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, int]:
    n = len(adjacency_list)
    lengths = np.fromiter((len(row) for row in adjacency_list), dtype=np.int64, count=n)

    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(lengths, out=indptr[1:])

    total_len = int(indptr[-1])
    indices = np.empty(total_len, dtype=np.int32)

    pos = 0
    for row in adjacency_list:
        row_len = len(row)
        if row_len:
            indices[pos : pos + row_len] = row
            pos += row_len

    return indptr, indices, total_len

def _global_efficiency_dense_scipy(
    indptr: np.ndarray, indices: np.ndarray, n: int
) -> float:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    data = np.ones(indices.shape[0], dtype=np.uint8)
    graph = csr_matrix((data, indices, indptr), shape=(n, n))
    dist = shortest_path(graph, directed=False, unweighted=True)

    inv = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv, where=dist > 0.0)
    return float(inv.sum() / (n * (n - 1)))

class Solver:
    def __init__(self) -> None:
        if njit is not None:
            dummy_indptr = np.array([0, 0], dtype=np.int64)
            dummy_indices = np.empty(0, dtype=np.int32)
            dummy_recip = np.array([0.0, 1.0], dtype=np.float64)
            _global_efficiency_csr_seq(dummy_indptr, dummy_indices, dummy_recip)
            _global_efficiency_csr_par(dummy_indptr, dummy_indices, dummy_recip)

    def solve(self, problem: dict[str, list[list[int]]], **kwargs: Any) -> Any:
        adjacency_list = problem["adjacency_list"]
        n = len(adjacency_list)

        if n <= 1:
            return {"global_efficiency": 0.0}

        indptr, indices, total_len = _build_csr_from_adjacency(adjacency_list)

        if total_len == 0:
            return {"global_efficiency": 0.0}

        if total_len == n * (n - 1):
            return {"global_efficiency": 1.0}

        max_edges = n * (n - 1) // 2
        undirected_edges_est = total_len >> 1

        if n <= 512 and undirected_edges_est * 6 >= max_edges:
            try:
                return {
                    "global_efficiency": _global_efficiency_dense_scipy(
                        indptr, indices, n
                    )
                }
            except Exception:
                pass

        recip = np.empty(n, dtype=np.float64)
        recip[0] = 0.0
        if n > 1:
            recip[1:] = 1.0 / np.arange(1, n, dtype=np.float64)

        if njit is not None and n >= 384 and total_len >= 4 * n:
            efficiency = _global_efficiency_csr_par(indptr, indices, recip)
        else:
            efficiency = _global_efficiency_csr_seq(indptr, indices, recip)

        return {"global_efficiency": float(efficiency)}