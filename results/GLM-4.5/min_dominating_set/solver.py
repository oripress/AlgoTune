from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        model = cp_model.CpModel()
        
        # Create boolean variables for each vertex
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        
        # Add domination constraints more efficiently
        for i in range(n):
            # For each vertex i, at least one of i or its neighbors must be selected
            # Use list comprehension for faster constraint creation
            constraint_vars = [nodes[i]]
            for j in range(n):
                if problem[i][j] == 1:
                    constraint_vars.append(nodes[j])
            model.AddBoolOr(constraint_vars)
        
        # Minimize the number of selected vertices
        model.Minimize(sum(nodes))
        
        # Solve with optimized parameters
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        solver.parameters.num_search_workers = 1
        solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
        solver.parameters.linearization_level = 1
        solver.parameters.cp_model_presolve = True
        solver.parameters.optimize_with_core = True
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            # Use list comprehension for faster extraction
            return [i for i in range(n) if solver.Value(nodes[i])]
        else:
            return []