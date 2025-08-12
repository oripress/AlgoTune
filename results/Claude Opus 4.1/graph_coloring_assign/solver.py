import networkx as nx
from networkx.algorithms.approximation import clique as approx_clique
from ortools.sat.python import cp_model
from itertools import combinations

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Optimized graph coloring solver.
        """
        n = len(problem)
        
        # Quick check for empty graph
        if n == 0:
            return []
        
        # Build NetworkX graph more efficiently
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j]:
                    edges.append((i, j))
        
        # Handle trivial case: no edges
        if not edges:
            return [1] * n
        
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        
        # Dominator preprocessing
        def coloring_preprocessing_fast(G_sub):
            dominator = {v: v for v in G_sub.nodes()}
            prev_size = -1
            max_iterations = 10  # Limit iterations to prevent timeout
            iteration = 0
            while len(G_sub.nodes()) != prev_size and iteration < max_iterations:
                prev_size = len(G_sub.nodes())
                adj = {v: set(G_sub.neighbors(v)) for v in G_sub.nodes()}
                redundant = []
                for u, v in combinations(G_sub.nodes(), 2):
                    if adj[u] <= adj[v]:
                        redundant.append(u)
                        dominator[u] = v
                    elif adj[v] <= adj[u]:
                        redundant.append(v)
                        dominator[v] = u
                G_sub.remove_nodes_from(redundant)
                iteration += 1
            return G_sub, dominator
        
        G_red, dominator = coloring_preprocessing_fast(G.copy())
        V = list(G_red.nodes())
        E = list(G_red.edges())
        
        # Handle case where reduction leaves empty graph
        if not V:
            greedy = nx.greedy_color(G, strategy="largest_first")
            return [greedy[i] + 1 for i in range(n)]
        
        # Upper bound via greedy
        greedy_red = nx.greedy_color(G_red)
        ub = len(set(greedy_red.values()))
        H = min(ub, n)  # Cap H at n
        
        # Heuristic best clique (with timeout protection)
        try:
            clique_set = approx_clique.max_clique(G_red)
            Q = sorted(clique_set)
            lb = len(Q)
        except:
            Q = []
            lb = 1
        
        # If clique size equals greedy bound, use greedy
        if lb == ub:
            greedy = nx.greedy_color(G, strategy="largest_first")
            return [greedy[i] + 1 for i in range(n)]
        
        # For very small reduced graphs, use greedy
        if len(V) <= 10:
            greedy = nx.greedy_color(G, strategy="largest_first")
            return [greedy[i] + 1 for i in range(n)]
        
        # Build CP-SAT model
        model = cp_model.CpModel()
        
        # Variables: x[u,i] = 1 if node u uses color i+1
        x = {}
        for u in V:
            for i in range(H):
                x[(u, i)] = model.NewBoolVar(f"x_{u}_{i}")
        
        # w[i] = 1 if color i+1 is used
        w = {}
        for i in range(H):
            w[i] = model.NewBoolVar(f"w_{i}")
        
        # Clique seeding: force each Q[i] to use a distinct color i+1
        for i in range(min(len(Q), H)):
            if i < len(Q):
                model.Add(x[(Q[i], i)] == 1)
        
        # Each vertex gets exactly one color
        for u in V:
            model.Add(sum(x[(u, i)] for i in range(H)) == 1)
        
        # Adjacent vertices cannot share the same color slot unless w[i]=1
        for u, v in E:
            for i in range(H):
                model.Add(x[(u, i)] + x[(v, i)] <= w[i])
        
        # Link w[i] to assignments
        for i in range(H):
            model.Add(w[i] <= sum(x[(u, i)] for u in V))
        
        # Symmetry breaking
        for i in range(1, H):
            model.Add(w[i - 1] >= w[i])
        
        # Objective: minimize number of colors used
        model.Minimize(sum(w[i] for i in range(H)))
        
        # Solve with time limit
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 0.5  # Tight time limit
        solver.parameters.num_search_workers = 1  # Single thread for consistency
        status = solver.Solve(model)
        
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            # Fallback to greedy
            greedy = nx.greedy_color(G, strategy="largest_first")
            return [greedy[i] + 1 for i in range(n)]
        
        # Extract assigned colors on reduced graph
        c_red = {}
        for u in V:
            for i in range(H):
                if solver.Value(x[(u, i)]) == 1:
                    c_red[u] = i + 1
                    break
        
        # Map back through dominator to original nodes
        colors = [0] * n
        for v in range(n):
            root = v
            max_hops = n  # Prevent infinite loop
            hops = 0
            while dominator.get(root, root) != root and hops < max_hops:
                root = dominator[root]
                hops += 1
            if root in c_red:
                colors[v] = c_red[root]
            else:
                colors[v] = 1  # Default color
        
        # Normalize so colors span 1..k
        used = sorted(set(colors))
        if 0 in used:
            used.remove(0)
        if not used:
            return [1] * n
        remap = {old: new for new, old in enumerate(used, start=1)}
        colors = [remap.get(c, 1) for c in colors]
        
        return colors