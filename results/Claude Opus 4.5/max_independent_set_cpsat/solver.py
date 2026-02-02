from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n == 0:
            return []
        
        # Build adjacency list efficiently
        adj = [[] for _ in range(n)]
        edges = []
        for i in range(n):
            row = problem[i]
            for j in range(i + 1, n):
                if row[j]:
                    edges.append((i, j))
                    adj[i].append(j)
                    adj[j].append(i)
        
        # If no edges, return all vertices
        if not edges:
            return list(range(n))
        
        # Fast greedy for hints
        selected = []
        excluded = [False] * n
        # Sort by degree (ascending) for greedy
        degree = [len(adj[i]) for i in range(n)]
        order = sorted(range(n), key=lambda x: degree[x])
        
        for v in order:
            if not excluded[v]:
                selected.append(v)
                for u in adj[v]:
                    excluded[u] = True
        
        greedy_set = set(selected)
        
        # Build CP model
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        
        # Add edge constraints
        for i, j in edges:
            model.Add(nodes[i] + nodes[j] <= 1)
        
        # Objective
        model.Maximize(sum(nodes))
        
        # Add hints from greedy
        for i in range(n):
            model.AddHint(nodes[i], 1 if i in greedy_set else 0)
        
        # Configure solver
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            return [i for i in range(n) if solver.Value(nodes[i]) == 1]
        return selected