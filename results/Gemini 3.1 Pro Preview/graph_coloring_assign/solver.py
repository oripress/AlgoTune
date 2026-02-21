from typing import Any
import pysat
from pysat.solvers import Cadical153

class Solver:
    def solve(self, problem: list[list[int]], **kwargs) -> Any:
        n = len(problem)
        if n == 0:
            return []

        # Dominator preprocessing using pure Python
        dominator = list(range(n))
        adj = [{j for j, val in enumerate(row) if val} for row in problem]
                    
        active_nodes = list(range(n))
        prev_size = -1
        while len(active_nodes) != prev_size:
            prev_size = len(active_nodes)
            redundant = set()
            for i in range(len(active_nodes)):
                u = active_nodes[i]
                if u in redundant:
                    continue
                adj_u = adj[u]
                for j in range(i + 1, len(active_nodes)):
                    v = active_nodes[j]
                    if v in redundant:
                        continue
                    if v not in adj_u:
                        adj_v = adj[v]
                        if adj_u <= adj_v:
                            redundant.add(u)
                            dominator[u] = v
                            break
                        elif adj_v <= adj_u:
                            redundant.add(v)
                            dominator[v] = u
            if redundant:
                active_nodes = [u for u in active_nodes if u not in redundant]
                for u in active_nodes:
                    adj[u] -= redundant

        V = active_nodes
        E = []
        for i in range(len(V)):
            u = V[i]
            adj_u = adj[u]
            for j in range(i + 1, len(V)):
                v = V[j]
                if v in adj_u:
                    E.append((u, v))

        # Greedy coloring (DSATUR-like)
        degrees = [len(adj[u]) for u in range(n)]
        color = {}
        adj_colors = {u: set() for u in V}
        uncolored = set(V)
        
        while uncolored:
            # Pick node with max dsat, then max degree
            u = max(uncolored, key=lambda x: (len(adj_colors[x]), degrees[x]))
            uncolored.remove(u)
            
            used_colors = adj_colors[u]
            c = 0
            while c in used_colors:
                c += 1
            color[u] = c
            
            for v in adj[u]:
                if v in uncolored:
                    adj_colors[v].add(c)
        ub = len(set(color.values()))

        # Heuristic best clique
        Q = []
        for u in sorted(V, key=lambda x: degrees[x], reverse=True):
            if all(v in adj[u] for v in Q):
                Q.append(u)
        lb = len(Q)

        if lb == ub:
            colors = [0] * n
            for v in range(n):
                root = v
                while dominator[root] != root:
                    root = dominator[root]
                colors[v] = color[root] + 1
            return colors

        V_idx = {u: i for i, u in enumerate(V)}
        E_idx = [(V_idx[u], V_idx[v]) for u, v in E]
        Q_idx = [V_idx[u] for u in Q]

        def solve_k(k):
            with Cadical153() as solver:
                clauses = []
                for i in range(len(V)):
                    clauses.append([i * k + c + 1 for c in range(k)])
                            
                for u_idx, v_idx in E_idx:
                    for c in range(k):
                        clauses.append([-(u_idx * k + c + 1), -(v_idx * k + c + 1)])
                        
                # Symmetry breaking: fix colors for the clique
                for i, u_idx in enumerate(Q_idx):
                    if i < k:
                        clauses.append([u_idx * k + i + 1])
                        
                solver.append_formula(clauses)
                
                if solver.solve():
                    model = solver.get_model()
                    c_red = {}
                    for i, u in enumerate(V):
                        for c in range(k):
                            if model[i * k + c] > 0:
                                c_red[u] = c + 1
                                break
                    return c_red
                return None

        best_c_red = {u: color[u] + 1 for u in V}
        
        # Linear search downwards
        k = ub - 1
        while k >= lb:
            res = solve_k(k)
            if res is not None:
                best_c_red = res
                k -= 1
            else:
                break

        colors = [0] * n
        for v in range(n):
            root = v
            while dominator[root] != root:
                root = dominator[root]
            colors[v] = best_c_red[root]
            
        used = sorted(set(colors))
        remap = {old: new for new, old in enumerate(used, start=1)}
        return [remap[c] for c in colors]