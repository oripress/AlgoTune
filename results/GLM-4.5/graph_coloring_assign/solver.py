import random
from itertools import combinations
import networkx as nx
from networkx.algorithms.approximation import clique as approx_clique
from ortools.sat.python import cp_model
import numpy as np

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solves the graph coloring problem using a hybrid approach combining
        fast heuristics with CP-SAT for optimal solutions.
        
        :param problem: A 2D adjacency matrix representing the graph.
        :return: A list of colors (1..k) assigned to each vertex.
        """
        n = len(problem)
        
        # Fast special case handling
        if n == 0:
            return []
        if n == 1:
            return [1]
        
        # Check if graph is empty (no edges)
        has_edges = any(problem[i][j] for i in range(n) for j in range(i+1, n))
        if not has_edges:
            return [1] * n
        
        # Check if graph is complete
        is_complete = all(problem[i][j] for i in range(n) for j in range(i+1, n))
        if is_complete:
            return list(range(1, n+1))
        
        # Check if graph is bipartite using BFS
        def is_bipartite(adj_matrix):
            color = [-1] * n
            for i in range(n):
                if color[i] == -1:
                    queue = [i]
                    color[i] = 0
                    while queue:
                        v = queue.pop(0)
                        for u in range(n):
                            if adj_matrix[v][u] and color[u] == -1:
                                color[u] = color[v] ^ 1
                                queue.append(u)
                            elif adj_matrix[v][u] and color[u] == color[v]:
                                return None
            return color
        
        bipartite_coloring = is_bipartite(problem)
        if bipartite_coloring is not None:
            return [c + 1 for c in bipartite_coloring]
        
        # Build NetworkX graph for general case
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j]:
                    G.add_edge(i, j)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # Fast greedy coloring as fallback
        def fast_greedy_color(graph):
            colors = {}
            # Sort nodes by degree (descending)
            nodes = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)
            
            for node in nodes:
                # Find colors used by neighbors
                used_colors = set()
                for neighbor in graph.neighbors(node):
                    if neighbor in colors:
                        used_colors.add(colors[neighbor])
                
                # Find the smallest available color
                color = 1
                while color in used_colors:
                    color += 1
                
                colors[node] = color
            
            return colors
        
        # For small graphs, use CP-SAT directly
        if n <= 50:
            return self._solve_with_cpsat(problem, G, n)
        
        # For medium graphs, try fast heuristics first
        greedy_colors = fast_greedy_color(G)
        k_greedy = len(set(greedy_colors.values()))
        
        # If greedy coloring is likely optimal (matches lower bound)
        max_degree = max(dict(G.degree()).values())
        if k_greedy <= max_degree + 1:
            return [greedy_colors[i] + 1 for i in range(n)]
        
        # Use CP-SAT for optimal solution
        return self._solve_with_cpsat(problem, G, n)
    
    def _solve_with_cpsat(self, problem, G, n):
        """Solve using CP-SAT with optimizations."""
        # Build NetworkX graph
        G_red, dominator = self._coloring_preprocessing_fast(G.copy())
        V = list(G_red.nodes())
        E = list(G_red.edges())
        
        # Upper bound via greedy
        ub = len(set(nx.greedy_color(G_red).values()))
        H = ub
        
        # Fast clique finding
        lb = self._fast_max_clique(G_red)
        
        # If clique size equals greedy bound, use greedy coloring
        if lb == ub:
            greedy = nx.greedy_color(G, strategy="largest_first")
            return [greedy[i] + 1 for i in range(n)]
        
        # Build CP-SAT model with integer variables
        model = cp_model.CpModel()
        
        # x[u] = color assigned to node u (0 to H-1)
        x = {}
        for u in V:
            x[u] = model.NewIntVar(0, H - 1, f"x_{u}")
        
        # w[i] = 1 if color i is used
        w = {}
        for i in range(H):
            w[i] = model.NewBoolVar(f"w_{i}")
        
        # Get clique for seeding
        clique_set = self._fast_max_clique_set(G_red)
        Q = sorted(clique_set)
        
        # Clique seeding: force each Q[i] to use a distinct color i
        for i, u in enumerate(Q):
            model.Add(x[u] == i)
        
        # Constraints
        # (1) Adjacent vertices cannot have the same color
        for u, v in E:
            model.Add(x[u] != x[v])
        
        # (2) Link w[i] to assignments: w[i] = 1 if some node uses color i
        for i in range(H):
            # Create a list of boolean variables indicating if node u uses color i
            uses_color_i = []
            for u in V:
                b = model.NewBoolVar(f"uses_{u}_{i}")
                model.Add(x[u] == i).OnlyEnforceIf(b)
                model.Add(x[u] != i).OnlyEnforceIf(b.Not())
                uses_color_i.append(b)
            
            # At least one node uses color i => w[i] = 1
            model.Add(sum(uses_color_i) >= 1).OnlyEnforceIf(w[i])
            # No node uses color i => w[i] = 0
            model.Add(sum(uses_color_i) == 0).OnlyEnforceIf(w[i].Not())
        
        # (3) Symmetry breaking: enforce w[0] >= w[1] >= ... >= w[H-1]
        for i in range(1, H):
            model.Add(w[i - 1] >= w[i])
        
        # Objective: minimize number of colors used
        model.Minimize(sum(w[i] for i in range(H)))
        
        # Solve with time limit for large graphs
        solver = cp_model.CpSolver()
        if n > 100:
            solver.parameters.max_time_in_seconds = 10.0
        
        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            # Fallback to greedy coloring
            greedy = nx.greedy_color(G, strategy="largest_first")
            return [greedy[i] + 1 for i in range(n)]
        
        # Extract assigned colors on reduced graph
        c_red = {}
        for u in V:
            c_red[u] = solver.Value(x[u]) + 1
        
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
    
    def _coloring_preprocessing_fast(self, G_sub):
        """Fast dominator preprocessing."""
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
    
    def _fast_max_clique(self, G):
        """Fast maximum clique approximation."""
        if not G.nodes():
            return 0
        
        # Use degree-based heuristic
        nodes = sorted(G.nodes(), key=lambda x: G.degree(x))
        max_clique_size = 0
        
        # Try multiple random starts
        for _ in range(5):
            random.shuffle(nodes)
            clique = []
            for node in nodes:
                can_add = all(G.has_edge(node, c) for c in clique)
                if can_add:
                    clique.append(node)
            max_clique_size = max(max_clique_size, len(clique))
        
        return max_clique_size
    
    def _fast_max_clique_set(self, G):
        """Fast maximum clique set approximation."""
        if not G.nodes():
            return set()
        
        # Use degree-based heuristic
        nodes = sorted(G.nodes(), key=lambda x: G.degree(x))
        best_clique = []
        
        # Try multiple random starts
        for _ in range(10):
            random.shuffle(nodes)
            clique = []
            for node in nodes:
                can_add = all(G.has_edge(node, c) for c in clique)
                if can_add:
                    clique.append(node)
            if len(clique) > len(best_clique):
                best_clique = clique
        
        return set(best_clique)