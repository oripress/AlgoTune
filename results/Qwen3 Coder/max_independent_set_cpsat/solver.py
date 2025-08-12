import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solves the max independent set problem using the CP-SAT solver with optimizations.
        
        :param problem: A 2d adjacency matrix representing the graph.
        :return: A list of node indices included in the maximum independent set.
        """
        n = len(problem)
        if n == 0:
            return []
        
        # Quick check for empty graph
        total_edges = sum(sum(row) for row in problem)
        if total_edges == 0:
            return list(range(n))
            
        # Preprocessing: identify isolated vertices (degree 0)
        isolated_vertices = []
        non_isolated_vertices = []
        for i in range(n):
            if sum(problem[i]) == 0:
                isolated_vertices.append(i)
            else:
                non_isolated_vertices.append(i)
        model = cp_model.CpModel()
        
        # Create a boolean variable for each vertex: 1 if included in the set, 0 otherwise.
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        
        # Add independence constraints: For every edge (i, j) in the graph,
        # at most one of the endpoints can be in the independent set.
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j] == 1:
                    model.Add(nodes[i] + nodes[j] <= 1)
        
        # Objective: Maximize the number of vertices chosen.
        model.Maximize(sum(nodes))
        
        # Solve the model with optimized parameters.
        solver = cp_model.CpSolver()
        # Set time limit for faster solving
        solver.parameters.max_time_in_seconds = 3.0
        # Enable presolve
        solver.parameters.cp_model_presolve = True
        # Use parallel solving
        solver.parameters.num_search_workers = 2
        # Focus on finding good solutions quickly
        solver.parameters.cp_model_probing_level = 1
        # Search strategy
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Extract and return nodes with value 1.
            selected = [i for i in range(n) if solver.Value(nodes[i]) == 1]
            return selected
        else:
            return []