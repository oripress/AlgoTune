from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0] if n == 1 else []

        model = cp_model.CpModel()
        x = {}
        obj_terms = []
        
        # Create variables and objective terms
        # We can iterate only once
        circuit_arcs = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    var = model.NewBoolVar('')
                    x[i, j] = var
                    obj_terms.append(problem[i][j] * var)
                    circuit_arcs.append((i, j, var))

        # Circuit constraint
        model.AddCircuit(circuit_arcs)
        
        # Minimize objective
        model.Minimize(sum(obj_terms))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        # Use single thread to avoid overhead for small instances
        solver.parameters.num_search_workers = 1
        solver.parameters.cp_model_presolve = False
        
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Extract edges
            next_node = {}
            for (i, j), var in x.items():
                if solver.Value(var):
                    next_node[i] = j
            
            path = [0]
            curr = 0
            for _ in range(n - 1):
                curr = next_node[curr]
                path.append(curr)
            path.append(0)
            return path
        return []