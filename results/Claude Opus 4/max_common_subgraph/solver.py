from ortools.sat.python import cp_model
import numpy as np

class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> list[tuple[int, int]]:
        """Find the maximum common subgraph between two graphs."""
        A = problem["A"]
        B = problem["B"]
        n, m = len(A), len(B)
        
        # Convert to numpy for faster access
        A_np = np.array(A)
        B_np = np.array(B)
        
        model = cp_model.CpModel()
        
        # x[i][p] = 1 if node i in G is mapped to node p in H
        x = [[model.NewBoolVar(f"x_{i}_{p}") for p in range(m)] for i in range(n)]
        
        # One-to-one mapping constraints
        for i in range(n):
            model.Add(sum(x[i][p] for p in range(m)) <= 1)
        for p in range(m):
            model.Add(sum(x[i][p] for i in range(n)) <= 1)
        
        # Edge consistency constraints - optimize by only adding necessary constraints
        # Only add constraints when edges differ
        for i in range(n):
            for j in range(i + 1, n):
                if A_np[i,j] == 1:  # Edge exists in G
                    for p in range(m):
                        for q in range(p + 1, m):
                            if B_np[p,q] == 0:  # No edge in H
                                model.Add(x[i][p] + x[j][q] <= 1)
                                model.Add(x[i][q] + x[j][p] <= 1)
                else:  # No edge in G
                    for p in range(m):
                        for q in range(p + 1, m):
                            if B_np[p,q] == 1:  # Edge exists in H
                                model.Add(x[i][p] + x[j][q] <= 1)
                                model.Add(x[i][q] + x[j][p] <= 1)
        
        # Objective: maximize size of the mapping
        model.Maximize(sum(x[i][p] for i in range(n) for p in range(m)))
        
        # Optimize solver parameters
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.linearization_level = 2
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            return [(i, p) for i in range(n) for p in range(m) if solver.Value(x[i][p]) == 1]
        else:
            return []