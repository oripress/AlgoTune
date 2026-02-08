from typing import Any
import numpy as np
from numba import njit


@njit(cache=True)
def solve_mcmf(n, s, t, heads, tails, caps, costs, num_edges):
    """Solve min cost max flow using successive shortest path with SPFA."""
    # Build adjacency list
    deg = np.zeros(n, dtype=np.int64)
    for i in range(num_edges):
        deg[tails[i]] += 1
    
    adj_offset = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        adj_offset[i + 1] = adj_offset[i] + deg[i]
    
    adj = np.zeros(num_edges, dtype=np.int64)
    pos = np.zeros(n, dtype=np.int64)
    for i in range(num_edges):
        u = tails[i]
        adj[adj_offset[u] + pos[u]] = i
        pos[u] += 1
    
    # Residual capacities
    res_cap = caps.copy()
    
    # Flow on each edge
    flow = np.zeros(num_edges, dtype=np.int64)
    
    # SPFA arrays
    dist = np.zeros(n, dtype=np.int64)
    in_queue = np.zeros(n, dtype=np.bool_)
    prev_edge = np.zeros(n, dtype=np.int64)
    
    INF = np.int64(10**18)
    
    # Queue for SPFA (circular buffer)
    queue = np.zeros(n + 1, dtype=np.int64)
    
    while True:
        # SPFA from s to t
        for i in range(n):
            dist[i] = INF
            in_queue[i] = False
            prev_edge[i] = -1
        
        dist[s] = 0
        in_queue[s] = True
        q_head = 0
        q_tail = 0
        queue[q_tail] = s
        q_tail = (q_tail + 1) % (n + 1)
        q_size = 1
        
        while q_size > 0:
            u = queue[q_head]
            q_head = (q_head + 1) % (n + 1)
            q_size -= 1
            in_queue[u] = False
            
            for idx in range(adj_offset[u], adj_offset[u + 1]):
                e = adj[idx]
                if res_cap[e] > 0:
                    v = heads[e]
                    new_dist = dist[u] + costs[e]
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        prev_edge[v] = e
                        if not in_queue[v]:
                            in_queue[v] = True
                            queue[q_tail] = v
                            q_tail = (q_tail + 1) % (n + 1)
                            q_size += 1
        
        if dist[t] == INF:
            break
        
        # Find bottleneck
        bottleneck = INF
        v = t
        while v != s:
            e = prev_edge[v]
            if res_cap[e] < bottleneck:
                bottleneck = res_cap[e]
            v = tails[e]
        
        # Augment flow
        v = t
        while v != s:
            e = prev_edge[v]
            res_cap[e] -= bottleneck
            res_cap[e ^ 1] += bottleneck
            flow[e] += bottleneck
            flow[e ^ 1] -= bottleneck
            v = tails[e]
    
    return flow


# Warm up numba
_dummy_heads = np.zeros(2, dtype=np.int64)
_dummy_tails = np.zeros(2, dtype=np.int64)
_dummy_caps = np.zeros(2, dtype=np.int64)
_dummy_costs = np.zeros(2, dtype=np.int64)
_dummy_heads[0] = 1
_dummy_tails[0] = 0
_dummy_heads[1] = 0
_dummy_tails[1] = 1
_dummy_caps[0] = 1
solve_mcmf(2, 0, 1, _dummy_heads, _dummy_tails, _dummy_caps, _dummy_costs, 2)


class Solver:
    def solve(self, problem, **kwargs) -> Any:
        capacity = problem["capacity"]
        cost_matrix = problem["cost"]
        s = problem["s"]
        t = problem["t"]
        n = len(capacity)
        
        # Build edge list
        # For each directed edge (i,j), we add forward and backward edges
        # Forward: (i->j, cap, cost) at index 2k
        # Backward: (j->i, 0, -cost) at index 2k+1
        
        # Count edges first
        edge_count = 0
        for i in range(n):
            for j in range(n):
                if capacity[i][j] > 0:
                    edge_count += 1
        
        num_edges = edge_count * 2
        if num_edges == 0:
            return [[0] * n for _ in range(n)]
        
        tails = np.empty(num_edges, dtype=np.int64)
        heads = np.empty(num_edges, dtype=np.int64)
        caps = np.empty(num_edges, dtype=np.int64)
        costs = np.empty(num_edges, dtype=np.int64)
        
        edge_map = []  # Maps original edge index to (i, j)
        idx = 0
        for i in range(n):
            for j in range(n):
                c = capacity[i][j]
                if c > 0:
                    w = cost_matrix[i][j]
                    # Forward edge at index idx
                    tails[idx] = i
                    heads[idx] = j
                    caps[idx] = int(c)
                    costs[idx] = int(w)
                    # Backward edge at index idx+1
                    tails[idx + 1] = j
                    heads[idx + 1] = i
                    caps[idx + 1] = 0
                    costs[idx + 1] = -int(w)
                    edge_map.append((i, j))
                    idx += 2
        
        flow = solve_mcmf(n, s, t, heads, tails, caps, costs, num_edges)
        
        solution = [[0] * n for _ in range(n)]
        for k in range(edge_count):
            f = flow[2 * k]
            if f > 0:
                i, j = edge_map[k]
                solution[i][j] = int(f)
        
        return solution