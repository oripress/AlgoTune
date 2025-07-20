import numpy as np
from numba import njit, types

@njit(types.int64[:](types.int32, types.int32[:, :]), nogil=True, fastmath=True, cache=True, boundscheck=False)
def find_articulation_points(num_nodes, edges):
    # Precompute neighbor lists using arrays
    degs = np.zeros(num_nodes, dtype=np.int32)
    for i in range(edges.shape[0]):
        u = edges[i, 0]
        v = edges[i, 1]
        degs[u] += 1
        degs[v] += 1
    
    # Build graph using cumulative sum
    graph_ptr = np.zeros(num_nodes + 1, dtype=np.int32)
    for i in range(1, num_nodes + 1):
        graph_ptr[i] = graph_ptr[i-1] + degs[i-1]
    graph_data = np.zeros(graph_ptr[num_nodes], dtype=np.int32)
    
    # Reset degs for indexing
    degs.fill(0)
    for i in range(edges.shape[0]):
        u = edges[i, 0]
        v = edges[i, 1]
        pos_u = graph_ptr[u] + degs[u]
        graph_data[pos_u] = v
        degs[u] += 1
        pos_v = graph_ptr[v] + degs[v]
        graph_data[pos_v] = u
        degs[v] += 1
    
    # Initialize arrays
    disc = np.full(num_nodes, -1, dtype=np.int32)
    low = np.full(num_nodes, -1, dtype=np.int32)
    ap = np.zeros(num_nodes, dtype=np.bool_)
    parent = np.full(num_nodes, -1, dtype=np.int32)
    next_index = np.zeros(num_nodes, dtype=np.int32)
    stack = np.zeros(num_nodes, dtype=np.int32)
    stack_ptr = 0
    time = 0
    
    # Use local references for performance
    graph_ptr_local = graph_ptr
    graph_data_local = graph_data
    
    for root in range(num_nodes):
        if disc[root] != -1:
            continue
            
        disc[root] = time
        low[root] = time
        time += 1
        stack[stack_ptr] = root
        stack_ptr += 1
        parent[root] = -1
        next_index[root] = graph_ptr_local[root]
        
        children = 0
        
        while stack_ptr > 0:
            u = stack[stack_ptr-1]
            start = next_index[u]
            end = graph_ptr_local[u+1]
            
            if start < end:
                v = graph_data_local[start]
                next_index[u] = start + 1
                
                if v == parent[u]:
                    continue
                    
                if disc[v] == -1:
                    disc[v] = time
                    low[v] = time
                    time += 1
                    stack[stack_ptr] = v
                    stack_ptr += 1
                    parent[v] = u
                    next_index[v] = graph_ptr_local[v]
                    if u == root:
                        children += 1
                else:
                    # Update low[u] using back edge
                    if low[u] > disc[v]:
                        low[u] = disc[v]
            else:
                stack_ptr -= 1
                pu = parent[u]
                if pu != -1:
                    # Use local variables for performance
                    low_u = low[u]
                    disc_pu = disc[pu]
                    
                    # Update parent's low
                    if low[pu] > low_u:
                        low[pu] = low_u
                    
                    # Check articulation point condition
                    if pu != root and low_u >= disc_pu:
                        ap[pu] = True
        
        # Root node condition
        if children >= 2:
            ap[root] = True

    return np.where(ap)[0].astype(np.int64)

class Solver:
    def solve(self, problem, **kwargs):
        num_nodes = problem["num_nodes"]
        edges = np.array(problem["edges"], dtype=np.int32)
        ap_arr = find_articulation_points(num_nodes, edges)
        ap_list = ap_arr.tolist()
        return {"articulation_points": ap_list}