import numpy as np
from numba import njit
from numba.typed import List
from typing import Any

@njit(cache=True)
def spfa_mcmf_sparse(n, edges, head, next_edge, s, t):
    """SPFA-based min cost max flow with edge list representation"""
    INF = 1e18
    num_edges = len(edges)
    
    # edges[i] = (to, cap, cost, rev_idx)
    edge_to = edges[:, 0].astype(np.int64)
    edge_cap = edges[:, 1].copy()
    edge_cost = edges[:, 2].copy()
    edge_rev = edges[:, 3].astype(np.int64)
    
    total_flow = 0.0
    
    while True:
        # SPFA for shortest path
        dist = np.full(n, INF, dtype=np.float64)
        dist[s] = 0.0
        parent_edge = np.full(n, -1, dtype=np.int64)
        in_queue = np.zeros(n, dtype=np.bool_)
        
        queue = np.empty(n * n + n, dtype=np.int64)
        qhead, qtail = 0, 0
        queue[qtail] = s
        qtail += 1
        in_queue[s] = True
        
        while qhead < qtail:
            u = queue[qhead]
            qhead += 1
            in_queue[u] = False
            
            e = head[u]
            while e != -1:
                v = edge_to[e]
                if edge_cap[e] > 1e-9:
                    nd = dist[u] + edge_cost[e]
                    if nd < dist[v] - 1e-9:
                        dist[v] = nd
                        parent_edge[v] = e
                        if not in_queue[v]:
                            queue[qtail] = v
                            qtail += 1
                            in_queue[v] = True
                e = next_edge[e]
        
        if dist[t] > INF / 2:
            break
        
        # Find bottleneck
        bottleneck = INF
        v = t
        while v != s:
            e = parent_edge[v]
            if edge_cap[e] < bottleneck:
                bottleneck = edge_cap[e]
            v = edge_to[edge_rev[e]]
        
        # Update flow
        v = t
        while v != s:
            e = parent_edge[v]
            edge_cap[e] -= bottleneck
            edge_cap[edge_rev[e]] += bottleneck
            v = edge_to[edge_rev[e]]
        
        total_flow += bottleneck
    
    return edge_to, edge_cap, edge_rev

@njit(cache=True)
def build_graph(capacity, cost_matrix, n):
    """Build edge list representation"""
    # Count edges
    edge_count = 0
    for i in range(n):
        for j in range(n):
            if capacity[i, j] > 0:
                edge_count += 1
    
    # Each original edge creates 2 edges (forward + reverse)
    num_edges = edge_count * 2
    edges = np.zeros((num_edges, 4), dtype=np.float64)
    head = np.full(n, -1, dtype=np.int64)
    next_edge = np.full(num_edges, -1, dtype=np.int64)
    edge_from = np.zeros(num_edges, dtype=np.int64)
    
    idx = 0
    for i in range(n):
        for j in range(n):
            if capacity[i, j] > 0:
                # Forward edge
                edges[idx] = (j, capacity[i, j], cost_matrix[i, j], idx + 1)
                edge_from[idx] = i
                next_edge[idx] = head[i]
                head[i] = idx
                
                # Reverse edge
                edges[idx + 1] = (i, 0, -cost_matrix[i, j], idx)
                edge_from[idx + 1] = j
                next_edge[idx + 1] = head[j]
                head[j] = idx + 1
                
                idx += 2
    
    return edges, head, next_edge, edge_from

@njit(cache=True)
def spfa_dense(cap, cost, s, t, n):
    """Dense SPFA-based min cost max flow - optimized"""
    flow = np.zeros((n, n), dtype=np.float64)
    INF = 1e18
    
    while True:
        dist = np.full(n, INF, dtype=np.float64)
        dist[s] = 0.0
        parent = np.full(n, -1, dtype=np.int64)
        in_queue = np.zeros(n, dtype=np.bool_)
        
        # Circular queue
        queue = np.empty(n + 1, dtype=np.int64)
        qhead, qtail = 0, 1
        queue[0] = s
        in_queue[s] = True
        qsize = n + 1
        
        while qhead != qtail:
            u = queue[qhead]
            qhead = (qhead + 1) % qsize
            in_queue[u] = False
            
            for v in range(n):
                res_cap = cap[u, v] - flow[u, v]
                if res_cap > 1e-9:
                    nd = dist[u] + cost[u, v]
                    if nd < dist[v] - 1e-9:
                        dist[v] = nd
                        parent[v] = u
                        if not in_queue[v]:
                            queue[qtail] = v
                            qtail = (qtail + 1) % qsize
                            in_queue[v] = True
        
        if dist[t] > INF / 2:
            break
        
        bottleneck = INF
        v = t
        while v != s:
            u = parent[v]
            res = cap[u, v] - flow[u, v]
            if res < bottleneck:
                bottleneck = res
            v = u
        
        v = t
        while v != s:
            u = parent[v]
            flow[u, v] += bottleneck
            flow[v, u] -= bottleneck
            v = u
    
    return flow

class Solver:
    def __init__(self):
        # JIT compile warmup
        cap = np.array([[0.0, 1.0], [0.0, 0.0]])
        cost = np.array([[0.0, 1.0], [-1.0, 0.0]])
        spfa_dense(cap, cost, 0, 1, 2)
    
    def solve(self, problem: dict, **kwargs) -> list[list[Any]]:
        capacity = problem["capacity"]
        cost_matrix = problem["cost"]
        s = problem["s"]
        t = problem["t"]
        n = len(capacity)
        
        if n == 0:
            return []
        
        cap = np.array(capacity, dtype=np.float64)
        cost = np.zeros((n, n), dtype=np.float64)
        
        for i in range(n):
            for j in range(n):
                if cap[i, j] > 0:
                    cost[i, j] = cost_matrix[i][j]
                    cost[j, i] = -cost_matrix[i][j]
        
        flow = spfa_dense(cap, cost, s, t, n)
        
        result = np.maximum(flow, 0)
        return result.tolist()