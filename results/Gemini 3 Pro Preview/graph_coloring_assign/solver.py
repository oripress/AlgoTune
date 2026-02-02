import numpy as np
import networkx as nx
from numba import njit
try:
    from pysat.solvers import Glucose3
except ImportError:
    Glucose3 = None

@njit
def reduce_graph_numba(adj, n):
    active = np.ones(n, dtype=np.bool_)
    dominator = np.arange(n)
    
    changed = True
    while changed:
        changed = False
        nodes = []
        for i in range(n):
            if active[i]:
                nodes.append(i)
        
        count = len(nodes)
        if count == 0: break
        
        for i in range(count):
            u = nodes[i]
            if not active[u]: continue
            
            for j in range(i + 1, count):
                v = nodes[j]
                if not active[v]: continue
                
                if adj[u, v]: continue
                
                u_subset_v = True
                v_subset_u = True
                
                for k_idx in range(count):
                    k = nodes[k_idx]
                    if k == u or k == v: continue
                    
                    u_has = adj[u, k]
                    v_has = adj[v, k]
                    
                    if u_has and not v_has:
                        u_subset_v = False
                    if v_has and not u_has:
                        v_subset_u = False
                    
                    if not u_subset_v and not v_subset_u:
                        break
                
                if u_subset_v:
                    active[u] = False
                    dominator[u] = v
                    changed = True
                    break 
                elif v_subset_u:
                    active[v] = False
                    dominator[v] = u
                    changed = True
                    
    return active, dominator

@njit
def dsatur_numba(adj, nodes):
    n = len(nodes)
    colors = np.zeros(n, dtype=np.int32)
    degrees = np.zeros(n, dtype=np.int32)
    sat_deg = np.zeros(n, dtype=np.int32)
    neighbor_colors_mask = np.zeros((n, n + 1), dtype=np.bool_)
    
    for i in range(n):
        u = nodes[i]
        d = 0
        for j in range(n):
            if i == j: continue
            if adj[u, nodes[j]]:
                d += 1
        degrees[i] = d
        
    colored_count = 0
    while colored_count < n:
        best_idx = -1
        max_sat = -1
        max_deg = -1
        
        for i in range(n):
            if colors[i] != 0: continue
            s = sat_deg[i]
            d = degrees[i]
            if s > max_sat:
                max_sat = s
                max_deg = d
                best_idx = i
            elif s == max_sat:
                if d > max_deg:
                    max_deg = d
                    best_idx = i
        
        if best_idx == -1: break
        
        c = 1
        while neighbor_colors_mask[best_idx, c]:
            c += 1
        
        colors[best_idx] = c
        colored_count += 1
        
        u = nodes[best_idx]
        for j in range(n):
            if colors[j] == 0:
                v = nodes[j]
                if adj[u, v]:
                    if not neighbor_colors_mask[j, c]:
                        neighbor_colors_mask[j, c] = True
                        sat_deg[j] += 1
                        
    return colors

class Solver:
    def __init__(self):
        # Trigger compilation
        dummy_adj = np.zeros((2, 2), dtype=np.int32)
        reduce_graph_numba(dummy_adj, 2)
        dsatur_numba(dummy_adj, np.array([0, 1], dtype=np.int32))

    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        if n == 0: return []
        
        adj = np.array(problem, dtype=np.int32)
        active, dominator = reduce_graph_numba(adj, n)
        
        active_nodes = np.where(active)[0].astype(np.int32)
        m = len(active_nodes)
        
        dsatur_colors = dsatur_numba(adj, active_nodes)
        ub = 0
        for c in dsatur_colors:
            if c > ub: ub = c
            
        G_red = nx.Graph()
        G_red.add_nodes_from(active_nodes)
        for i in range(m):
            u = active_nodes[i]
            for j in range(i + 1, m):
                v = active_nodes[j]
                if adj[u, v]:
                    G_red.add_edge(u, v)
                    
        from networkx.algorithms.approximation import clique as approx_clique
        clique_set = approx_clique.max_clique(G_red)
        lb = len(clique_set)
        
        final_colors_map = {}
        
        if lb == ub:
            for i in range(m):
                final_colors_map[active_nodes[i]] = dsatur_colors[i]
        elif Glucose3 is None:
             for i in range(m):
                final_colors_map[active_nodes[i]] = dsatur_colors[i]
        else:
            node_map = {node: i for i, node in enumerate(active_nodes)}
            sat_edges = []
            for u, v in G_red.edges():
                sat_edges.append((node_map[u], node_map[v]))
            
            found = False
            for k in range(lb, ub):
                with Glucose3() as solver:
                    for u_idx in range(m):
                        solver.add_clause([(u_idx * k) + c + 1 for c in range(k)])
                        
                    for u_idx, v_idx in sat_edges:
                        for c in range(k):
                            v1 = (u_idx * k) + c + 1
                            v2 = (v_idx * k) + c + 1
                            solver.add_clause([-v1, -v2])
                            
                    clique_indices = [node_map[u] for u in clique_set if u in node_map]
                    for i, u_idx in enumerate(clique_indices):
                        if i < k:
                            solver.add_clause([(u_idx * k) + i + 1])

                    if solver.solve():
                        model = solver.get_model()
                        model_set = set(model)
                        for u_idx in range(m):
                            for c in range(k):
                                if ((u_idx * k) + c + 1) in model_set:
                                    final_colors_map[active_nodes[u_idx]] = c + 1
                                    break
                        found = True
                        break
            
            if not found:
                for i in range(m):
                    final_colors_map[active_nodes[i]] = dsatur_colors[i]

        colors = [0] * n
        for v in range(n):
            root = v
            while dominator[root] != root:
                root = dominator[root]
            colors[v] = final_colors_map[root]
            
        used = sorted(set(colors))
        remap = {old: new for new, old in enumerate(used, start=1)}
        return [remap[c] for c in colors]