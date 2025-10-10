import networkx as nx
from ortools.sat.python import cp_model
from itertools import combinations

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Optimized graph coloring solver using CP-SAT with preprocessing.
        """
        n = len(problem)
        
        # For very small graphs, use greedy directly
        if n <= 5:
            G = nx.Graph()
            G.add_nodes_from(range(n))
            for i in range(n):
                for j in range(i + 1, n):
                    if problem[i][j]:
                        G.add_edge(i, j)
            greedy = nx.greedy_color(G, strategy="largest_first")
            return [greedy[i] + 1 for i in range(n)]
        
        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if problem[i][j]]
        G.add_edges_from(edges)
        
        # Dominator preprocessing
        def coloring_preprocessing_fast(G_sub):
            dominator = {v: v for v in G_sub.nodes()}
            prev_size = -1
            while len(G_sub.nodes()) != prev_size:
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
            return G_sub, dominator
        
        G_red, dominator = coloring_preprocessing_fast(G.copy())
        V = list(G_red.nodes())
        E = list(G_red.edges())
        
        # Upper bound via greedy
        greedy_colors = nx.greedy_color(G_red, strategy="largest_first")
        ub = len(set(greedy_colors.values()))
        H = ub
        
        # Fast clique lower bound using greedy approach
        def greedy_clique(G_sub):
            """Fast greedy clique finder"""
            if len(G_sub.nodes()) == 0:
                return []
            
            # Start with node of highest degree
            degrees = dict(G_sub.degree())
            u = max(degrees, key=degrees.get)
            clique = [u]
            candidates = set(G_sub.neighbors(u))
            
            while candidates:
                # Find candidate with most connections to current clique
                best = None
                best_count = -1
                for v in candidates:
                    count = sum(1 for c in clique if G_sub.has_edge(v, c))
                    if count == len(clique) and (best is None or degrees[v] > degrees[best]):
                        best = v
                        best_count = count
                
                if best is None:
                    break
                    
                clique.append(best)
                candidates = candidates.intersection(set(G_sub.neighbors(best)))
            
            return clique
        
        Q = greedy_clique(G_red)
        lb = len(Q)
        
        # If clique size equals greedy bound, use greedy
        if lb == ub:
            greedy = nx.greedy_color(G, strategy="largest_first")
            return [greedy[i] + 1 for i in range(n)]
        
        # Build CP-SAT model
        model = cp_model.CpModel()
        
        # x[u,i] = 1 if node u uses color i+1
        x = {}
        for u in V:
            for i in range(H):
                x[(u, i)] = model.NewBoolVar(f"x_{u}_{i}")
        
        # w[i] = 1 if color i+1 is used
        w = {}
        for i in range(H):
            w[i] = model.NewBoolVar(f"w_{i}")
        
        # Clique seeding
        for i, u in enumerate(Q):
            model.Add(x[(u, i)] == 1)
        
        # Each vertex gets exactly one color
        for u in V:
            model.Add(sum(x[(u, i)] for i in range(H)) == 1)
        
        # Adjacent vertices cannot share the same color
        for u, v in E:
            for i in range(H):
                model.Add(x[(u, i)] + x[(v, i)] <= w[i])
        
        # Link w[i] to assignments
        for i in range(H):
            model.Add(w[i] <= sum(x[(u, i)] for u in V))
        
        # Symmetry breaking
        for i in range(1, H):
            model.Add(w[i - 1] >= w[i])
        
        # Minimize number of colors used
        model.Minimize(sum(w[i] for i in range(H)))
        
        # Solve with time limit
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            # Fallback to greedy if no optimal found
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
            while dominator[root] != root:
                root = dominator[root]
            colors[v] = c_red[root]
        
        # Normalize so colors span 1..k
        used = sorted(set(colors))
        remap = {old: new for new, old in enumerate(used, start=1)}
        colors = [remap[c] for c in colors]
        
        return colors