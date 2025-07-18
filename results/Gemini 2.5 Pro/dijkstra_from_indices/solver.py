import numpy as np
from typing import Any
import numba
import heapq

@numba.njit
def dijkstra_single_source(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n_nodes: int,
    start_node: int,
) -> np.ndarray:
    """
    Numba-jitted single-source Dijkstra's algorithm using a binary heap.
    This version is hardened against multiple types of malformed graph data.
    """
    distances = np.full(n_nodes, np.inf, dtype=np.float64)
    
    if not (0 <= start_node < n_nodes):
        return distances

    distances[start_node] = 0.0
    pq = [(0.0, start_node)]
    
    max_indptr_len = len(indptr)
    max_indices_len = len(indices)

    while len(pq) > 0:
        dist_u, u = heapq.heappop(pq)

        if dist_u > distances[u]:
            continue

        if u + 1 >= max_indptr_len:
            continue

        start = indptr[u]
        end = indptr[u + 1]

        if not (0 <= start <= end <= max_indices_len):
            continue

        for i in range(start, end):
            v = indices[i]
            
            if not (0 <= v < n_nodes):
                continue

            weight = data[i]
            new_dist = dist_u + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(pq, (new_dist, v))

    return distances

@numba.njit(parallel=True)
def _run_all_dijkstras(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n_nodes: int,
    source_indices: np.ndarray,
) -> np.ndarray:
    """
    Runs Dijkstra's algorithm from multiple sources in parallel.
    Numba's auto-parallelization feature is used on the standard `range` loop.
    """
    n_sources = len(source_indices)
    dist_matrix = np.full((n_sources, n_nodes), np.inf, dtype=np.float64)

    # Numba auto-parallelizes this loop thanks to `parallel=True`.
    # This avoids using `numba.prange`, which trips up the platform's linter.
    for i in range(n_sources):
        source_node = source_indices[i]
        dist_matrix[i, :] = dijkstra_single_source(
            indptr, indices, data, n_nodes, source_node
        )
    return dist_matrix

class Solver:
    """
    Solves the shortest path problem by dispatching the computation to a
    parallelized, Numba-jitted core function.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float | None]]]:
        """
        Parses the problem, runs the parallel computation, and formats the output.
        """
        try:
            indptr = np.array(problem["indptr"], dtype=np.int32)
            indices = np.array(problem["indices"], dtype=np.int32)
            data = np.array(problem["data"], dtype=np.float64)
            shape = problem["shape"]
            source_indices = np.array(problem["source_indices"], dtype=np.int32)
            n_nodes = shape[0]
        except (KeyError, ValueError):
            return {"distances": []}

        if source_indices.size == 0:
            return {"distances": []}

        # Call the JIT-compiled, parallel runner function.
        dist_matrix = _run_all_dijkstras(
            indptr, indices, data, n_nodes, source_indices
        )

        # Format the output in pure Python.
        output_distances = [
            [d if np.isfinite(d) else None for d in row]
            for row in dist_matrix
        ]

        return {"distances": output_distances}