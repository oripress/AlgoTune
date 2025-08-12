from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        A = problem["A"]
        B = problem["B"]
        n = len(A)
        m = len(B)
        model = cp_model.CpModel()
        
        # Precompute neighbors and non-neighbors as tuples for faster iteration
        neighbors_H = [tuple(q for q in range(m) if p != q and B[p][q] == 1) for p in range(m)]
        non_edges_H = [tuple(q for q in range(m) if p != q and B[p][q] == 0) for p in range(m)]
        
        # Create Boolean variables for node mappings
        x = [[model.NewBoolVar(f'x_{i}_{p}') for p in range(m)] for i in range(n)]
        
        # One-to-one mapping constraints
        for i in range(n):
            model.Add(sum(x[i]) <= 1)
        for p in range(m):
            model.Add(sum(x[i][p] for i in range(n)) <= 1)
        
        # Optimized constraint generation with local variables
        for i in range(n):
            row_i = x[i]  # Local variable for faster access
            for j in range(i + 1, n):
                row_j = x[j]  # Local variable for faster access
                if A[i][j] == 1:
                    for p in range(m):
                        non_edges = non_edges_H[p]
                        if non_edges:
                            # Forbid mappings to non-adjacent nodes
                            constraint_list = [row_i[p]]
                            constraint_list.extend(row_j[q] for q in non_edges)
                            model.AddAtMostOne(constraint_list)
                else:
                    for p in range(m):
                        neighbors = neighbors_H[p]
                        if neighbors:
                            # Forbid mappings to adjacent nodes
                            constraint_list = [row_i[p]]
                            constraint_list.extend(row_j[q] for q in neighbors)
                            model.AddAtMostOne(constraint_list)
        
        # Maximize the size of the mapping
        model.Maximize(sum(x[i][p] for i in range(n) for p in range(m)))
        
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.log_search_progress = False
        solver.parameters.symmetry_level = 1
        solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
        solver.parameters.linearization_level = 0
        solver.parameters.cp_model_presolve = True
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            return [(i, p) for i in range(n) for p in range(m) if solver.Value(x[i][p]) == 1]
        else:
            return []