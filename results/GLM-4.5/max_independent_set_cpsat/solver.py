from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs) -> list[int]:
        """
        Solves the max independent set problem using optimized CP-SAT solver
        with efficient constraint generation and tuned parameters.
        
        :param problem: A 2d adjacency matrix representing the graph.
        :return: A list of node indices included in the maximum independent set.
        """
        n = len(problem)
        model = cp_model.CpModel()
        
        # Create a boolean variable for each vertex: 1 if included in the set, 0 otherwise.
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        
        # Direct constraint generation without intermediate list
        # This is more memory efficient and faster for large graphs
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j] == 1:
                    model.Add(nodes[i] + nodes[j] <= 1)
        
        # Objective: Maximize the number of vertices chosen.
        model.Maximize(sum(nodes))
        
        # Solve the model with optimized parameters
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        solver.parameters.num_search_workers = 4  # Balanced threading
        solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH  # Let solver choose best strategy
        solver.parameters.linearization_level = 1
        solver.parameters.cp_model_presolve = True
        solver.parameters.cp_model_probing_level = 1
        solver.parameters.use_phase_saving = True
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            # Extract and return nodes with value 1.
            selected = [i for i in range(n) if solver.Value(nodes[i]) == 1]
            return selected
        else:
            return []