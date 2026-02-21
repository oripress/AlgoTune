import numpy as np
from numba import njit

@njit
def mwis_numba(adj, weights):
    n = len(weights)
    
    # Sort by weight descending to find good solutions early
    order = np.argsort(weights)[::-1]
    
    weights_reordered = np.zeros(n, dtype=np.float64)
    for i in range(n):
        weights_reordered[i] = weights[order[i]]
        
    # Build CSR-like adjacency list for forward edges
    head = np.zeros(n + 1, dtype=np.int32)
    edges = np.zeros(n * n, dtype=np.int32)
    edge_cnt = 0
    for i in range(n):
        head[i] = edge_cnt
        for j in range(i + 1, n):
            if adj[order[i], order[j]]:
                edges[edge_cnt] = j
                edge_cnt += 1
    head[n] = edge_cnt
            
    # Greedy initialization to find a good lower bound
    best_w = 0.0
    best_s = np.zeros(n, dtype=np.bool_)
    
    greedy_available = np.ones(n, dtype=np.bool_)
    for i in range(n):
        if greedy_available[i]:
            best_s[i] = True
            best_w += weights_reordered[i]
            for e in range(head[i], head[i + 1]):
                greedy_available[edges[e]] = False
                
    current_s = np.zeros(n, dtype=np.bool_)
    available = np.ones(n, dtype=np.bool_)
    
    modified = np.zeros((n, n), dtype=np.int32)
    num_modified = np.zeros(n, dtype=np.int32)
    
    node = 0
    current_w = 0.0
    
    state = np.zeros(n + 1, dtype=np.int32) 
    rem_w_arr = np.zeros(n + 1, dtype=np.float64)
    rem_w_arr[0] = np.sum(weights_reordered)
    
    while node >= 0:
        if state[node] == 0:
            if current_w > best_w:
                best_w = current_w
                best_s[:] = current_s[:]
                
            if node == n:
                node -= 1
                continue
                
            if current_w + rem_w_arr[node] <= best_w:
                state[node] = 0
                node -= 1
                continue
                
            if available[node]:
                # Branch 1: Include node
                current_s[node] = True
                current_w += weights_reordered[node]
                
                delta_w = 0.0
                count = 0
                for e in range(head[node], head[node + 1]):
                    i = edges[e]
                    if available[i]:
                        available[i] = False
                        modified[node, count] = i
                        count += 1
                        delta_w += weights_reordered[i]
                num_modified[node] = count
                
                rem_w_arr[node + 1] = rem_w_arr[node] - weights_reordered[node] - delta_w
                state[node] = 1
                node += 1
            else:
                # Node not available, only one branch: Exclude node
                rem_w_arr[node + 1] = rem_w_arr[node]
                state[node] = 2
                node += 1
                
        elif state[node] == 1:
            # Backtracked from Include branch. Now do Exclude branch.
            current_s[node] = False
            current_w -= weights_reordered[node]
            
            count = num_modified[node]
            for i in range(count):
                available[modified[node, i]] = True
                
            rem_w_arr[node + 1] = rem_w_arr[node] - weights_reordered[node]
            state[node] = 2
            node += 1
            
        elif state[node] == 2:
            # Backtracked from Exclude branch.
            state[node] = 0
            node -= 1

    result = np.zeros(n, dtype=np.int64)
    count = 0
    for i in range(n):
        if best_s[i]:
            result[count] = order[i]
            count += 1
    return result[:count]

class Solver:
    def solve(self, problem: dict, **kwargs) -> list:
        adj_matrix = np.array(problem["adj_matrix"], dtype=np.bool_)
        weights = np.array(problem["weights"], dtype=np.float64)
        
        res = mwis_numba(adj_matrix, weights)
        return res.tolist()