import numpy as np
from numba import njit

@njit(cache=True)
def dijkstra_numba(data, indices, indptr, n, source_indices):
    distances = np.full(n, np.inf)
    
    max_heap_size = len(data) + len(source_indices)
    heap_nodes = np.empty(max_heap_size, dtype=np.int32)
    heap_dists = np.empty(max_heap_size, dtype=np.float64)
    heap_size = 0
    
    for i in range(len(source_indices)):
        src = source_indices[i]
        distances[src] = 0.0
        # push
        idx = heap_size
        heap_size += 1
        heap_nodes[idx] = src
        heap_dists[idx] = 0.0
        # sift up 4-ary
        while idx > 0:
            parent = (idx - 1) >> 2
            if heap_dists[idx] < heap_dists[parent]:
                tmp_node = heap_nodes[idx]
                heap_nodes[idx] = heap_nodes[parent]
                heap_nodes[parent] = tmp_node
                
                tmp_dist = heap_dists[idx]
                heap_dists[idx] = heap_dists[parent]
                heap_dists[parent] = tmp_dist
                
                idx = parent
            else:
                break
                
    while heap_size > 0:
        # pop
        u = heap_nodes[0]
        d = heap_dists[0]
        
        heap_size -= 1
        if heap_size > 0:
            heap_nodes[0] = heap_nodes[heap_size]
            heap_dists[0] = heap_dists[heap_size]
            
            # sift down 4-ary
            idx = 0
            while True:
                child1 = (idx << 2) + 1
                if child1 >= heap_size:
                    break
                
                smallest = idx
                min_dist = heap_dists[idx]
                
                # Unroll the loop for the 4 children
                if heap_dists[child1] < min_dist:
                    smallest = child1
                    min_dist = heap_dists[child1]
                    
                child2 = child1 + 1
                if child2 < heap_size and heap_dists[child2] < min_dist:
                    smallest = child2
                    min_dist = heap_dists[child2]
                    
                child3 = child1 + 2
                if child3 < heap_size and heap_dists[child3] < min_dist:
                    smallest = child3
                    min_dist = heap_dists[child3]
                    
                child4 = child1 + 3
                if child4 < heap_size and heap_dists[child4] < min_dist:
                    smallest = child4
                    min_dist = heap_dists[child4]
                    
                if smallest != idx:
                    tmp_node = heap_nodes[idx]
                    heap_nodes[idx] = heap_nodes[smallest]
                    heap_nodes[smallest] = tmp_node
                    
                    tmp_dist = heap_dists[idx]
                    heap_dists[idx] = heap_dists[smallest]
                    heap_dists[smallest] = tmp_dist
                    
                    idx = smallest
                else:
                    break
                    
        if d > distances[u]:
            continue
            
        start = indptr[u]
        end = indptr[u+1]
        for i in range(start, end):
            v = indices[i]
            weight = data[i]
            new_dist = d + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                # push
                idx = heap_size
                heap_size += 1
                heap_nodes[idx] = v
                heap_dists[idx] = new_dist
                # sift up 4-ary
                while idx > 0:
                    parent = (idx - 1) >> 2
                    if heap_dists[idx] < heap_dists[parent]:
                        tmp_node = heap_nodes[idx]
                        heap_nodes[idx] = heap_nodes[parent]
                        heap_nodes[parent] = tmp_node
                        
                        tmp_dist = heap_dists[idx]
                        heap_dists[idx] = heap_dists[parent]
                        heap_dists[parent] = tmp_dist
                        
                        idx = parent
                    else:
                        break
                        
    return distances

class Solver:
    def __init__(self, **kwargs):
        # Trigger Numba compilation with dummy data
        dummy_data = np.array([1.0], dtype=np.float64)
        dummy_indices = np.array([0], dtype=np.int32)
        dummy_indptr = np.array([0, 1], dtype=np.int32)
        dummy_sources = np.array([0], dtype=np.int32)
        dijkstra_numba(dummy_data, dummy_indices, dummy_indptr, 1, dummy_sources)

    def solve(self, problem, **kwargs):
        data = np.asarray(problem["data"], dtype=np.float64)
        indices = np.asarray(problem["indices"], dtype=np.int32)
        indptr = np.asarray(problem["indptr"], dtype=np.int32)
        source_indices = np.asarray(problem["source_indices"], dtype=np.int32)
        n = problem["shape"][0]
        
        dist_matrix = dijkstra_numba(data, indices, indptr, n, source_indices)
        
        dist_matrix = dist_matrix[np.newaxis, :]
        is_inf = np.isinf(dist_matrix)
        dist_matrix_obj = dist_matrix.astype(object)
        dist_matrix_obj[is_inf] = None
            
        return {"distances": dist_matrix_obj.tolist()}