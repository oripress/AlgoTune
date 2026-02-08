import numpy as np
from typing import Any
import heapq
from numba import njit, prange

@njit(cache=True)
def dijkstra_single(indptr, indices, data, n, source):
    """Single-source Dijkstra."""
    dist = np.full(n, np.inf)
    dist[source] = 0.0
    heap = [(0.0, source)]
    while len(heap) > 0:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for j in range(indptr[u], indptr[u + 1]):
            v = indices[j]
            new_dist = d + data[j]
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    return dist

@njit(parallel=True, cache=True)
def dijkstra_multi(indptr, indices, data, n, sources):
    """Multi-source Dijkstra in parallel."""
    num_sources = len(sources)
    result = np.empty((num_sources, n))
    for i in prange(num_sources):
        result[i] = dijkstra_single(indptr, indices, data, n, sources[i])
    return result

@njit(cache=True)
def dijkstra_multi_source_min(indptr, indices, data, n, sources):
    """Multi-source Dijkstra: min distance from any source to each node."""
    dist = np.full(n, np.inf)
    heap = [(0.0, sources[0])]
    dist[sources[0]] = 0.0
    for k in range(1, len(sources)):
        s = sources[k]
        dist[s] = 0.0
        heapq.heappush(heap, (0.0, s))
    
    while len(heap) > 0:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for j in range(indptr[u], indptr[u + 1]):
            v = indices[j]
            new_dist = d + data[j]
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    return dist

@njit(cache=True)
def dijkstra_single_undirected(indptr, indices, data, indptr_t, indices_t, data_t, n, source):
    """Single-source Dijkstra for undirected graph using forward+reverse."""
    dist = np.full(n, np.inf)
    dist[source] = 0.0
    heap = [(0.0, source)]
    while len(heap) > 0:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for j in range(indptr[u], indptr[u + 1]):
            v = indices[j]
            new_dist = d + data[j]
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
        for j in range(indptr_t[u], indptr_t[u + 1]):
            v = indices_t[j]
            new_dist = d + data_t[j]
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    return dist

@njit(cache=True)
def dijkstra_multi_source_min_undirected(indptr, indices, data, indptr_t, indices_t, data_t, n, sources):
    """Multi-source Dijkstra for undirected graph: min distance from any source."""
    dist = np.full(n, np.inf)
    heap = [(0.0, sources[0])]
    dist[sources[0]] = 0.0
    for k in range(1, len(sources)):
        s = sources[k]
        dist[s] = 0.0
        heapq.heappush(heap, (0.0, s))
    
    while len(heap) > 0:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for j in range(indptr[u], indptr[u + 1]):
            v = indices[j]
            new_dist = d + data[j]
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
        for j in range(indptr_t[u], indptr_t[u + 1]):
            v = indices_t[j]
            new_dist = d + data_t[j]
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    return dist

@njit(cache=True)
def compute_transpose_csr(indptr, indices, data, n):
    """Compute transpose of CSR matrix."""
    nnz = len(data)
    count = np.zeros(n, dtype=np.int64)
    for i in range(nnz):
        count[indices[i]] += 1
    
    indptr_t = np.empty(n + 1, dtype=np.int64)
    indptr_t[0] = 0
    for i in range(n):
        indptr_t[i + 1] = indptr_t[i] + count[i]
    
    indices_t = np.empty(nnz, dtype=np.int64)
    data_t = np.empty(nnz, dtype=np.float64)
    pos = np.zeros(n, dtype=np.int64)
    
    for i in range(n):
        for j in range(indptr[i], indptr[i + 1]):
            col = indices[j]
            dest = indptr_t[col] + pos[col]
            indices_t[dest] = i
            data_t[dest] = data[j]
            pos[col] += 1
    
    return indptr_t, indices_t, data_t

@njit(cache=True)
def symmetrize_csr(indptr, indices, data, n):
    """Create symmetric CSR by combining forward and backward edges."""
    nnz = len(data)
    
    # Count edges per row in both directions
    row_count = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j_idx in range(indptr[i], indptr[i + 1]):
            j = indices[j_idx]
            row_count[i] += 1
            row_count[j] += 1
    
    # Build intermediate CSR
    new_indptr = np.empty(n + 1, dtype=np.int64)
    new_indptr[0] = 0
    for i in range(n):
        new_indptr[i + 1] = new_indptr[i] + row_count[i]
    
    total = new_indptr[n]
    new_indices = np.empty(total, dtype=np.int64)
    new_data = np.empty(total, dtype=np.float64)
    
    pos = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j_idx in range(indptr[i], indptr[i + 1]):
            j = indices[j_idx]
            w = data[j_idx]
            # Forward edge
            dest = new_indptr[i] + pos[i]
            new_indices[dest] = j
            new_data[dest] = w
            pos[i] += 1
            # Backward edge
            dest = new_indptr[j] + pos[j]
            new_indices[dest] = i
            new_data[dest] = w
            pos[j] += 1
    
    # Sort each row by column index and deduplicate (min weight)
    final_count = np.zeros(n, dtype=np.int64)
    for i in range(n):
        start = new_indptr[i]
        length = row_count[i]
        end = start + length
        
        # Insertion sort by column index
        for a in range(start + 1, end):
            key_col = new_indices[a]
            key_w = new_data[a]
            b = a - 1
            while b >= start and new_indices[b] > key_col:
                new_indices[b + 1] = new_indices[b]
                new_data[b + 1] = new_data[b]
                b -= 1
            new_indices[b + 1] = key_col
            new_data[b + 1] = key_w
        
        # Deduplicate
        if length > 0:
            write = start
            for read in range(start + 1, end):
                if new_indices[read] == new_indices[write]:
                    if new_data[read] < new_data[write]:
                        new_data[write] = new_data[read]
                else:
                    write += 1
                    new_indices[write] = new_indices[read]
                    new_data[write] = new_data[read]
            final_count[i] = write - start + 1
        else:
            final_count[i] = 0
    
    # Compact
    sym_indptr = np.empty(n + 1, dtype=np.int64)
    sym_indptr[0] = 0
    for i in range(n):
        sym_indptr[i + 1] = sym_indptr[i] + final_count[i]
    
    sym_nnz = sym_indptr[n]
    sym_indices = np.empty(sym_nnz, dtype=np.int64)
    sym_data = np.empty(sym_nnz, dtype=np.float64)
    
    for i in range(n):
        src = new_indptr[i]
        dst = sym_indptr[i]
        for k in range(final_count[i]):
            sym_indices[dst + k] = new_indices[src + k]
            sym_data[dst + k] = new_data[src + k]
    
    return sym_indptr, sym_indices, sym_data


def _convert_row_to_list(dist):
    """Convert 1D distance array to list with None for inf."""
    if dist.ndim > 1:
        dist = dist[0]
    row = dist.tolist()
    inf_val = float('inf')
    has_inf = False
    for i in range(len(row)):
        if row[i] == inf_val:
            row[i] = None
            has_inf = True
    return [row]


class Solver:
    def __init__(self):
        indptr = np.array([0, 2, 4], dtype=np.int64)
        indices = np.array([0, 1, 0, 1], dtype=np.int64)
        data = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        sources = np.array([0, 1], dtype=np.int64)
        
        # Warm up all numba functions
        indptr_t, indices_t, data_t = compute_transpose_csr(indptr, indices, data, 2)
        dijkstra_single_undirected(indptr, indices, data, indptr_t, indices_t, data_t, 2, np.int64(0))
        dijkstra_multi_source_min_undirected(indptr, indices, data, indptr_t, indices_t, data_t, 2, sources)
        
        sym_indptr, sym_indices, sym_data = symmetrize_csr(indptr, indices, data, 2)
        dijkstra_single(sym_indptr, sym_indices, sym_data, 2, np.int64(0))
        dijkstra_multi_source_min(sym_indptr, sym_indices, sym_data, 2, sources)
    
    def solve(self, problem: dict, **kwargs) -> Any:
        n = problem["shape"][0]
        source_indices = problem["source_indices"]
        if not source_indices:
            return {"distances": []}

        num_sources = len(source_indices)
        
        d = problem["data"]
        if isinstance(d, np.ndarray):
            data_np = d if d.dtype == np.float64 else d.astype(np.float64)
        else:
            data_np = np.array(d, dtype=np.float64)
        
        idx = problem["indices"]
        if isinstance(idx, np.ndarray):
            indices_np = idx if idx.dtype == np.int64 else idx.astype(np.int64)
        else:
            indices_np = np.array(idx, dtype=np.int64)
        
        ip = problem["indptr"]
        if isinstance(ip, np.ndarray):
            indptr_np = ip if ip.dtype == np.int64 else ip.astype(np.int64)
        else:
            indptr_np = np.array(ip, dtype=np.int64)
        
        sources = np.array(source_indices, dtype=np.int64)
        
        # Compute transpose for undirected traversal
        indptr_t, indices_t, data_t = compute_transpose_csr(indptr_np, indices_np, data_np, n)
        
        if num_sources == 1:
            dist = dijkstra_single_undirected(
                indptr_np, indices_np, data_np,
                indptr_t, indices_t, data_t,
                n, sources[0]
            )
        else:
            # Min-only approach for multiple sources
            dist = dijkstra_multi_source_min_undirected(
                indptr_np, indices_np, data_np,
                indptr_t, indices_t, data_t,
                n, sources
            )

        return {"distances": _convert_row_to_list(dist)}