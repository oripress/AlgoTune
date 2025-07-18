import networkx as nx
from ortools.sat.python import cp_model
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n == 0:
            return []

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j]:
                    G.add_edge(i, j)
                    
        # Find connected components
        csr_graph = csr_matrix(problem)
        n_components, labels = connected_components(csr_graph, directed=False)
        if n_components > 1:
            colors = [0] * n
            # Group nodes by component
            comp_nodes = defaultdict(list)
            for i, lbl in enumerate(labels):
                comp_nodes[lbl].append(i)
                
            # Solve each component separately
            for lbl, nodes in comp_nodes.items():
                # Build subgraph for this component
                subgraph = G.subgraph(nodes).copy()
                comp_colors = self.solve_component(subgraph, nodes)
                for i, node in enumerate(nodes):
                    colors[node] = comp_colors[i]
            return colors
            
        return self.solve_component(G, list(range(n)))
    
    def solve_component(self, G, nodes):
        n = len(nodes)
        if n == 0:
            return []
            
        # Skip dominator preprocessing for large graphs
        if n > 100:
            V = list(range(n))
            E = list(G.edges())
            dominator = {i: i for i in range(n)}
        else:
            # Optimized dominator preprocessing
            G_red = G.copy()
            dominator = {v: v for v in G_red.nodes()}
            changed = True
            while changed:
                changed = False
                adj = {v: set(G_red.neighbors(v)) for v in G_red.nodes()}
                redundant = []
                for u in list(G_red.nodes()):
                    if u not in adj: continue
                    for v in list(G_red.nodes()):
                        if u == v or v not in adj: continue
                        if adj[u].issubset(adj[v]):
                            redundant.append(u)
                            dominator[u] = v
                            changed = True
                            break
                if redundant:
                    G_red.remove_nodes_from(redundant)
            V = list(G_red.nodes())
            E = list(G_red.edges())

        # Upper bound via DSATUR
        coloring = nx.coloring.greedy_color(G, strategy='DSATUR')
        ub = max(coloring.values()) + 1
        H = min(ub, n)  # Ensure H doesn't exceed n

        # Find max clique
        clique_set = set()
        if n <= 100:  # Use exact for small graphs
            for clique in nx.find_cliques(G):
                if len(clique) > len(clique_set):
                    clique_set = set(clique)
        else:
            clique_set = nx.approximation.max_clique(G)
        Q = sorted(clique_set)
        lb = len(Q)

        # If clique size equals greedy bound, use greedy coloring
        if lb == ub:
            return [coloring[i] + 1 for i in range(n)]

        # Build optimized CP-SAT model
        model = cp_model.CpModel()
        
        # Limit colors to lb + 20 to reduce variables
        max_colors = min(H, lb + 20)
        x = {}
        for u in V:
            for i in range(max_colors):
                x[(u, i)] = model.NewBoolVar(f"x_{u}_{i}")
        w = [model.NewBoolVar(f"w_{i}") for i in range(max_colors)]

        # Clique seeding
        for i, u in enumerate(Q):
            if i < max_colors:
                model.Add(x[(u, i)] == 1)

        # Constraints
        for u in V:
            model.Add(sum(x[(u, i)] for i in range(max_colors)) == 1)
            
        for u, v in E:
            for i in range(max_colors):
                model.Add(x[(u, i)] + x[(v, i)] <= w[i])
                
        for i in range(max_colors):
            model.Add(w[i] <= sum(x[(u, i)] for u in V))
            
        # Improved symmetry breaking
        model.Add(w[0] == 1)
        for i in range(1, max_colors):
            model.Add(w[i-1] >= w[i])

        # Objective
        model.Minimize(sum(w))

        # Solve with parallel execution
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 4
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)
        
        if status != cp_model.OPTIMAL:
            # Fallback to DSATUR coloring
            return [coloring[i] + 1 for i in range(n)]

        # Extract solution
        c_red = {}
        for u in V:
            for i in range(max_colors):
                if solver.Value(x[(u, i)]):
                    c_red[u] = i + 1
                    break

        # Map back through dominator
        colors = [0] * n
        for i, node in enumerate(nodes):
            root = i
            while dominator[root] != root:
                root = dominator[root]
            colors[i] = c_red.get(root, 1)

        # Normalize colors
        used_colors = sorted(set(colors))
        remap = {old: new for new, old in enumerate(used_colors, start=1)}
        return [remap[c] for c in colors]