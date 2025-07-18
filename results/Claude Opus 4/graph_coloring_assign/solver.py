import networkx as nx
from itertools import combinations
from networkx.algorithms.approximation import clique as approx_clique
from ortools.sat.python import cp_model
import numpy as np

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solves the graph coloring problem using optimized techniques.
        
        :param problem: A 2D adjacency matrix representing the graph.
        :return: A list of colors (1..k) assigned to each vertex.
        """
        n = len(problem)
        
        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j]:
                    G.add_edge(i, j)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # Dominator preprocessing
        G_red, dominator = self._coloring_preprocessing_fast(G.copy())
        V = list(G_red.nodes())
        E = list(G_red.edges())
        
        # Upper bound via greedy
        greedy_result = nx.greedy_color(G_red, strategy="largest_first")
        ub = len(set(greedy_result.values()))
        
        # Lower bound via clique
        clique_set = approx_clique.max_clique(G_red)
        Q = sorted(clique_set)
        lb = len(Q)
        
        # If clique size equals greedy bound, use greedy coloring
        if lb == ub:
            greedy = nx.greedy_color(G, strategy="largest_first")
            return [greedy[i] + 1 for i in range(n)]
        
        # Use CP-SAT for optimal solution
        H = ub  # number of color slots
        model = cp_model.CpModel()
        
        # Variables
        x = {}
        for u in V:
            for i in range(H):
                x[(u, i)] = model.NewBoolVar(f"x_{u}_{i}")
        
        w = {}
        for i in range(H):
            w[i] = model.NewBoolVar(f"w_{i}")
        
        # Clique seeding
        for i, u in enumerate(Q):
            if i < H:
                model.Add(x[(u, i)] == 1)
        
        # Constraints
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
        
        # Objective
        model.Minimize(sum(w[i] for i in range(H)))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0  # Add time limit
        status = solver.Solve(model)
        
        if status != cp_model.OPTIMAL:
            # Fallback to greedy if no optimal found
            greedy = nx.greedy_color(G, strategy="largest_first")
            return [greedy[i] + 1 for i in range(n)]
        
        # Extract solution
        c_red = {}
        for u in V:
            for i in range(H):
                if solver.Value(x[(u, i)]) == 1:
                    c_red[u] = i + 1
                    break
        
        # Map back through dominator
        colors = [0] * n
        for v in range(n):
            root = v
            while dominator[root] != root:
                root = dominator[root]
            colors[v] = c_red[root]
        
        # Normalize colors
        used = sorted(set(colors))
        remap = {old: new for new, old in enumerate(used, start=1)}
        colors = [remap[c] for c in colors]
        
        return colors
    
    def _coloring_preprocessing_fast(self, G_sub):
        """Dominator preprocessing to reduce graph size."""
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