class Solver:
    def solve(self, problem):
        """
        Solves the max independent set problem.
        
        :param problem: A 2d adjacency matrix representing the graph.
        :return: A list of node indices included in the maximum independent set.
        """
        n = len(problem)
        
        # For now, let's implement the reference solution to get started
        from ortools.sat.python import cp_model
        
        model = cp_model.CpModel()
        
        # Create a boolean variable for each vertex
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        
        # Add independence constraints
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j] == 1:
                    model.Add(nodes[i] + nodes[j] <= 1)
        
        # Objective: Maximize the number of vertices chosen
        model.Maximize(sum(nodes))
        
        # Solve the model
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            selected = [i for i in range(n) if solver.Value(nodes[i]) == 1]
            return selected
        else:
            return []