import networkx as nx
from networkx.algorithms.approximation import clique as approx_clique
from ortools.sat.python import cp_model
from typing import Any
from itertools import combinations

class Solver:
    def _dsatur_coloring(self, G):
        nodes = list(G.nodes())
        if not nodes:
            return {}
        
        adj = {v: set(G.neighbors(v)) for v in nodes}
        degrees = {v: len(adj[v]) for v in nodes}
        saturation = {v: set() for v in nodes}
        
        uncolored_nodes = set(nodes)
        colors = {}
        
        if not uncolored_nodes:
            return {}

        start_node = max(degrees, key=degrees.get)
        
        uncolored_nodes.remove(start_node)
        colors[start_node] = 0
        for neighbor in adj[start_node]:
            if neighbor in saturation:
                saturation[neighbor].add(0)

        while uncolored_nodes:
            node_to_color = max(
                uncolored_nodes, 
                key=lambda u: (len(saturation[u]), degrees[u])
            )

            used_neighbor_colors = {colors[n] for n in adj[node_to_color] if n in colors}
            
            color = 0
            while color in used_neighbor_colors:
                color += 1
            
            colors[node_to_color] = color
            uncolored_nodes.remove(node_to_color)
            
            for neighbor in adj[node_to_color]:
                if neighbor in uncolored_nodes:
                    saturation[neighbor].add(color)
                    
        return colors

    def solve(self, problem: list[list[int]], **kwargs) -> Any:
        n = len(problem)
        if n == 0:
            return []

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j]:
                    G.add_edge(i, j)
        
        def coloring_preprocessing_fast(G_orig):
            nodes = list(G_orig.nodes())
            dominator = {v: v for v in nodes}
            adj = {v: set(G_orig.neighbors(v)) for v in nodes}
            
            active_nodes = set(nodes)

            while True:
                nodes_to_scan = sorted(list(active_nodes), key=lambda n: len(adj[n]))
                redundant = set()
                
                for i in range(len(nodes_to_scan)):
                    for j in range(i + 1, len(nodes_to_scan)):
                        u, v = nodes_to_scan[i], nodes_to_scan[j]

                        if u in redundant or v in redundant:
                            continue

                        if adj[u].issubset(adj[v]):
                            redundant.add(u)
                            dominator[u] = v
                
                if not redundant:
                    break
                
                active_nodes.difference_update(redundant)
                
                for r_node in redundant:
                    neighbors = adj.pop(r_node)
                    for neighbor in neighbors:
                        if neighbor in adj:
                            adj[neighbor].discard(r_node)

            G_red = nx.Graph()
            if active_nodes:
                G_red.add_nodes_from(adj.keys())
                for u, neighbors in adj.items():
                    for v in neighbors:
                        if u < v:
                            G_red.add_edge(u, v)

            return G_red, dominator

        G_red, dominator = coloring_preprocessing_fast(G)
        V = list(G_red.nodes())
        E = list(G_red.edges())

        final_map = {}

        if nx.is_bipartite(G_red):
            if not E:
                final_map = {v: 0 for v in V}
            else:
                bipartite_sets = nx.bipartite.sets(G_red)
                for node in bipartite_sets[0]:
                    final_map[node] = 0
                for node in bipartite_sets[1]:
                    final_map[node] = 1
        else:
            best_heuristic_map = self._dsatur_coloring(G_red)
            ub = len(set(best_heuristic_map.values())) if best_heuristic_map else 0
            if ub == 0 and len(V) > 0:
                ub = 1
            H = ub

            clique_set = approx_clique.max_clique(G_red)
            Q = sorted(list(clique_set))
            lb = len(Q)

            if lb > H: H = lb
            
            if lb == ub:
                final_map = best_heuristic_map
            else:
                model = cp_model.CpModel()
                x = {(u, i): model.NewBoolVar(f"x_{u}_{i}") for u in V for i in range(H)}
                
                for u in V:
                    model.AddExactlyOne(x[(u, i)] for i in range(H))

                for u, v in E:
                    for i in range(H):
                        model.Add(x[(u, i)] + x[(v, i)] <= 1)

                w = [model.NewBoolVar(f"w_{i}") for i in range(H)]
                for i in range(H):
                    model.AddMaxEquality(w[i], [x[(u, i)] for u in V])

                for i in range(H - 1):
                    model.Add(w[i] >= w[i+1])
                
                model.Add(sum(w) >= lb)

                for i, u in enumerate(Q):
                    model.Add(x[(u, i)] == 1)

                model.Minimize(sum(w))

                for u in V:
                    if u in best_heuristic_map:
                        color = best_heuristic_map[u]
                        if color < H:
                            model.AddHint(x[(u, color)], 1)

                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = 0.4
                status = solver.Solve(model)

                if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                    for u in V:
                        for i in range(H):
                            if solver.Value(x[(u, i)]) == 1:
                                final_map[u] = i
                                break
                else:
                    final_map = best_heuristic_map

        colors = [0] * n
        for i in range(n):
            root = i
            while dominator[root] != root:
                root = dominator[root]
            if root in final_map:
                colors[i] = final_map[root]
            else:
                colors[i] = 0

        used = sorted(list(set(colors)))
        remap = {old: new for new, old in enumerate(used, start=1)}
        return [remap[c] for c in colors]