from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        """Optimized MDS solver with preprocessing"""
        n = len(problem)
        if n == 0:
            return []
        
        # Preprocessing: handle isolated nodes
        degrees = [sum(row) for row in problem]
        isolated = [i for i in range(n) if degrees[i] == 0]
        non_isolated = [i for i in range(n) if degrees[i] > 0]
        
        # If entire graph is isolated, return all nodes
        if not non_isolated:
            return isolated
        
        # Build subgraph for non-isolated nodes using list comprehensions
        m = len(non_isolated)
        idx_map = {node: idx for idx, node in enumerate(non_isolated)}
        sub_adj = [[0] * m for _ in range(m)]
        for i, node_i in enumerate(non_isolated):
            for j, node_j in enumerate(non_isolated):
                if problem[node_i][node_j]:
                    sub_adj[i][j] = 1
        
        # Solve subgraph with CP-SAT
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(m)]
        
        # Add constraints
        for i in range(m):
            neighbors = [nodes[i]]
            for j in range(m):
                if sub_adj[i][j] == 1:
                    neighbors.append(nodes[j])
            model.Add(sum(neighbors) >= 1)
        
        # Objective: minimize nodes
        model.Minimize(sum(nodes))
        
        # Configure solver
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 4  # Parallel search
        solver.parameters.log_search_progress = False
        
        # Solve
        status = solver.Solve(model)
        
        # Extract solution
        if status == cp_model.OPTIMAL:
            sub_solution = [j for j in range(m) if solver.Value(nodes[j]) == 1]
            solution_non_isolated = [non_isolated[j] for j in sub_solution]
            return isolated + solution_non_isolated
        return []