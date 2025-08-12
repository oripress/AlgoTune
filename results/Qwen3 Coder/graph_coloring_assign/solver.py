import networkx as nx
from ortools.sat.python import cp_model
from itertools import combinations
import numpy as np
import numba
from scipy import sparse
import networkx as nx
from ortools.sat.python import cp_model
from itertools import combinations

class Solver:
    def solve(self, problem, **kwargs):
        """Solve graph coloring problem with optimized approach"""
        n = len(problem)

        # Quick checks for simple cases
        if n == 0:
            return []
        if n == 1:
            return [1]

        # Convert to numpy for faster operations
        adj_matrix = np.array(problem)

        # Check if graph has no edges
        if np.count_nonzero(adj_matrix) == 0:
            return [1] * n

        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j] == 1:
                    G.add_edge(i, j)
        G.remove_edges_from(nx.selfloop_edges(G))

        # Preprocessing - remove dominated vertices
        G_red, dominator = self.coloring_preprocessing_fast(G.copy())
        V = list(G_red.nodes())
        E = list(G_red.edges())

        # Upper bound via greedy
        ub = len(set(nx.greedy_color(G_red, strategy="largest_first").values()))

        # Lower bound via simple heuristic (faster than max_clique)
        lb = self.simple_clique_lower_bound(G_red)

        # If bounds match, use greedy solution
        if lb == ub:
            greedy = nx.greedy_color(G, strategy="largest_first")
            return [greedy[i] + 1 for i in range(n)]

        # Use CP-SAT for optimization with tighter bounds
        return self.cp_sat_optimized(G_red, G, dominator, lb, ub, n)

    def simple_clique_lower_bound(self, G):
        """Fast approximation of clique lower bound"""
        if len(G.nodes()) == 0:
            return 0

        # Use a better greedy approach
        nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
        if not nodes:
            return 1

        # Find a maximal clique using a greedy approach
        clique = [nodes[0]]  # Start with highest degree node
        candidates = set(G.neighbors(nodes[0]))
        
        # Greedily add nodes to the clique
        for node in nodes[1:]:
            # Check if node can be added to current clique
            if all(G.has_edge(node, c) for c in clique):
                clique.append(node)
        
        return len(clique)

    def coloring_preprocessing_fast(self, G):
        """Fast preprocessing to remove dominated vertices"""
        dominator = {v: v for v in G.nodes()}
        prev_size = -1

        # Limit iterations to avoid excessive preprocessing
        iterations = 0
        max_iterations = 2

        while len(G.nodes()) != prev_size and iterations < max_iterations:
            prev_size = len(G.nodes())
            adj = {v: set(G.neighbors(v)) for v in G.nodes()}
            redundant = []

            # Check pairs for domination (limit pairs checked)
            nodes = list(G.nodes())
            max_pairs = min(100, len(nodes) * (len(nodes) - 1) // 2)  # Limit pairs
            pair_count = 0

            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if pair_count >= max_pairs:
                        break
                    u, v = nodes[i], nodes[j]
                    if u in adj and v in adj:  # Check if both still exist
                        if adj[u] <= adj[v] and len(adj[u]) < len(adj[v]):
                            redundant.append(u)
                            dominator[u] = v
                        elif adj[v] <= adj[u] and len(adj[v]) < len(adj[u]):
                            redundant.append(v)
                            dominator[v] = u
                    pair_count += 1
                if pair_count >= max_pairs:
                    break

            G.remove_nodes_from(redundant)
            iterations += 1

        return G, dominator

    def cp_sat_optimized(self, G_red, G_orig, dominator, lb, ub, n):
        """Optimized CP-SAT model for graph coloring"""
        V = list(G_red.nodes())
        E = list(G_red.edges())
        H = ub  # number of color slots

        # If reduced graph is empty
        if not V:
            colors = [1] * n
            # Apply dominator mapping for consistency
            for v in range(n):
                root = v
                while dominator[root] != root:
                    root = dominator[root]
                # All get same color since no constraints
            return colors

        # Build CP-SAT model using assignment formulation like reference
        model = cp_model.CpModel()

        # x[u,i] = 1 if node u uses color i+1
        x = {}
        for u in V:
            for i in range(H):
                x[(u, i)] = model.NewBoolVar(f"x_{u}_{i}")

        # w[i] = 1 if color i+1 is used by at least one vertex
        w = {}
        for i in range(H):
            w[i] = model.NewBoolVar(f"w_{i}")

        # Clique seeding: find a good clique to seed
        Q = []
        if len(V) > 0:
            # Simple clique heuristic
            nodes_sorted = sorted(V, key=lambda x: len([n for n in G_red.neighbors(x)]), reverse=True)
            if nodes_sorted:
                max_node = nodes_sorted[0]
                neighbors = list(G_red.neighbors(max_node))
                Q = [max_node]
                for node in neighbors:
                    if all(G_red.has_edge(node, q) for q in Q):
                        Q.append(node)
                # Limit clique size to avoid over-constraining
                Q = Q[:min(len(Q), H)]

        # Clique seeding: force each Q[i] to use a distinct color i+1
        for i, u in enumerate(Q):
            if i < H:  # Make sure we don't exceed available colors
                model.Add(x[(u, i)] == 1)

        # Constraints
        # (1) Each vertex gets exactly one color
        for u in V:
            model.Add(sum(x[(u, i)] for i in range(H)) == 1)

        # (2) Adjacent vertices cannot share the same color slot unless w[i]=1
        for u, v in E:
            for i in range(H):
                model.Add(x[(u, i)] + x[(v, i)] <= w[i])

        # (3) Link w[i] to assignments: if w[i]=1 then some x[u,i]=1
        for i in range(H):
            model.Add(w[i] <= sum(x[(u, i)] for u in V))

        # (4) Symmetry breaking: enforce w[0] >= w[1] >= ... >= w[H-1]
        for i in range(1, H):
            model.Add(w[i - 1] >= w[i])

        # Objective: minimize number of colors used
        model.Minimize(sum(w[i] for i in range(H)))

        # Solve with time limit
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 1.0  # Reduced time limit

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
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
        else:
            # Fallback to greedy with better strategy
            greedy = nx.greedy_color(G_orig, strategy="largest_first")
            return [greedy[i] + 1 for i in range(n)]