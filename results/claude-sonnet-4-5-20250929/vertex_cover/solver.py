from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solves the minimum vertex cover problem using OR-Tools CP-SAT.
        
        :param problem: a 2d-array (adj matrix)
        :return: A list indicating the selected nodes forming minimum vertex cover
        """
        
        try:
            n = len(problem)
            
            # Pre-compute edges
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if problem[i][j] == 1:
                        edges.append((i, j))
            
            if not edges:
                return []
            
            # Create CP-SAT model
            model = cp_model.CpModel()
            
            # Create binary variables for each vertex (avoid string formatting)
            x = [model.NewBoolVar('') for _ in range(n)]
            
            # Vertex cover constraints: for each edge, at least one endpoint must be in cover
            for i, j in edges:
                model.Add(x[i] + x[j] >= 1)
            
            # Minimize the number of vertices in the cover
            model.Minimize(sum(x))
            
            # Solve the model with optimized parameters
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 10.0
            solver.parameters.num_search_workers = 8
            solver.parameters.log_search_progress = False
            status = solver.Solve(model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                return [i for i in range(n) if solver.Value(x[i]) == 1]
            
            return list(range(n))
            
        except Exception as e:
            return list(range(len(problem)))