import numpy as np
from numba import njit
from typing import Any

@njit(fastmath=True)
def min_heap_push_or_update(heap_dist, heap_node, pos, size, dist, node):
    idx = pos[node]
    if idx == -1:
        i = size
        size += 1
    else:
        if dist >= heap_dist[idx]:
            return size
        i = idx
        
    while i > 0:
        p = (i - 1) // 2
        if heap_dist[p] <= dist:
            break
        p_node = heap_node[p]
        heap_dist[i] = heap_dist[p]
        heap_node[i] = p_node
        pos[p_node] = i
        i = p
        
    heap_dist[i] = dist
    heap_node[i] = node
    pos[node] = i
    return size

@njit(fastmath=True)
def min_heap_pop(heap_dist, heap_node, pos, size):
    ret_dist = heap_dist[0]
    ret_node = heap_node[0]
    pos[ret_node] = -1
    size -= 1
    if size > 0:
        dist = heap_dist[size]
        node = heap_node[size]
        i = 0
        while i * 2 + 1 < size:
            left = i * 2 + 1
            right = i * 2 + 2
            smallest = left
            if right < size and heap_dist[right] < heap_dist[left]:
                smallest = right
            if dist <= heap_dist[smallest]:
                break
            s_node = heap_node[smallest]
            heap_dist[i] = heap_dist[smallest]
            heap_node[i] = s_node
            pos[s_node] = i
            i = smallest
        heap_dist[i] = dist
        heap_node[i] = node
        pos[node] = i
    return ret_dist, ret_node, size

from numba import prange

@njit(parallel=True, fastmath=True)
def all_pairs_dijkstra(n, indptr, indices, data):
    dist_matrix = np.full((n, n), np.inf, dtype=np.float64)
    
    for i in prange(n):  # pylint: disable=not-an-iterable
        dist_matrix[i, i] = 0.0
        heap_dist = np.empty(n, dtype=np.float64)
        heap_node = np.empty(n, dtype=np.int32)
        pos = np.full(n, -1, dtype=np.int32)
        
        heap_size = 0
        heap_size = min_heap_push_or_update(heap_dist, heap_node, pos, heap_size, 0.0, i)
        
        while heap_size > 0:
            d, u, heap_size = min_heap_pop(heap_dist, heap_node, pos, heap_size)
            
            for j in range(indptr[u], indptr[u+1]):
                v = indices[j]
                weight = data[j]
                new_dist = d + weight
                if new_dist < dist_matrix[i, v]:
                    dist_matrix[i, v] = new_dist
                    heap_size = min_heap_push_or_update(heap_dist, heap_node, pos, heap_size, new_dist, v)
    return dist_matrix

class Solver:
    def __init__(self):
        # Trigger Numba compilation
        dummy_indptr = np.array([0, 1, 2], dtype=np.int32)
        dummy_indices = np.array([1, 0], dtype=np.int32)
        dummy_data = np.array([1.0, 1.0], dtype=np.float64)
        all_pairs_dijkstra(2, dummy_indptr, dummy_indices, dummy_data)

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[list[float]]]:
        try:
            n = problem["shape"][0]
            data = np.array(problem["data"], dtype=np.float64)
            indices = np.array(problem["indices"], dtype=np.int32)
            indptr = np.array(problem["indptr"], dtype=np.int32)
            
            dist_matrix = all_pairs_dijkstra(n, indptr, indices, data)
            dist_matrix_list = dist_matrix.tolist()

            return {"distance_matrix": dist_matrix_list}
            return {"distance_matrix": dist_matrix_list}
        except Exception:
            return {"distance_matrix": []}