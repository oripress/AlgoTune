import networkx as nx
from networkx.algorithms.approximation import clique as approx_clique
from ortools.sat.python import cp_model
from itertools import combinations

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n == 0:
            return []
        
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j]:
                    G.add_edge(i, j)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        if G.number_of_edges() == 0:
            return [1] * n
        
        # Dominator preprocessing
        def preprocess(G_sub):
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
        
        G_red, dominator = preprocess(G.copy())
        V = list(G_red.nodes())
        E = list(G_red.edges())
        
        if len(V) == 0:
            return [1] * n
        
        ub = len(set(nx.greedy_color(G_red).values()))
        Q = sorted(approx_clique.max_clique(G_red))
        lb = len(Q)
        
        if lb == ub:
            greedy = nx.greedy_color(G, strategy="largest_first")
            return [greedy[i] + 1 for i in range(n)]
        
        # CP-SAT model
        H = ub
        model = cp_model.CpModel()
        
        x = {(u, i): model.NewBoolVar(f"x_{u}_{i}") for u in V for i in range(H)}
        w = {i: model.NewBoolVar(f"w_{i}") for i in range(H)}
        
        for i, u in enumerate(Q):
            model.Add(x[(u, i)] == 1)
        
        for u in V:
            model.Add(sum(x[(u, i)] for i in range(H)) == 1)
        
        for u, v in E:
            for i in range(H):
                model.Add(x[(u, i)] + x[(v, i)] <= w[i])
        
        for i in range(H):
            model.Add(w[i] <= sum(x[(u, i)] for u in V))
        
        for i in range(1, H):
            model.Add(w[i - 1] >= w[i])
        
        model.Minimize(sum(w[i] for i in range(H)))
        
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        
        if status != cp_model.OPTIMAL:
            return []
        
        c_red = {}
        for u in V:
            for i in range(H):
                if solver.Value(x[(u, i)]) == 1:
                    c_red[u] = i + 1
                    break
        
        colors = [0] * n
        for v in range(n):
            root = v
            while dominator[root] != root:
                root = dominator[root]
            colors[v] = c_red[root]
        
        used = sorted(set(colors))
        remap = {old: new for new, old in enumerate(used, start=1)}
        return [remap[c] for c in colors]