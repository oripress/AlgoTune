from ortools.sat.python import cp_model
from typing import Any

class Solver:
    def solve(self, problem: dict[str, list[list[int]]], **kwargs) -> Any:
        A = problem["A"]
        B = problem["B"]
        n, m = len(A), len(B)
        
        # Quick check for empty graphs
        if n == 0 or m == 0:
            return []
        
        model = cp_model.CpModel()
        
        # x[i][p] = 1 if node i in G is mapped to node p in H
        x = [[model.NewBoolVar(f"x_{i}_{p}") for p in range(m)] for i in range(n)]
        
        # One-to-one mapping constraints
        for i in range(n):
            model.Add(sum(x[i][p] for p in range(m)) <= 1)
        for p in range(m):
            model.Add(sum(x[i][p] for i in range(n)) <= 1)
        
        # Edge consistency constraints (optimized)
        for i in range(n):
            for j in range(i + 1, n):
                for p in range(m):
                    for q in range(p + 1, m):  # Only check p < q
                        if A[i][j] != B[p][q]:
                            model.Add(x[i][p] + x[j][q] <= 1)
                            model.Add(x[i][q] + x[j][p] <= 1)  # Symmetric case
        
        # Objective: maximize size of the mapping
        model.Maximize(sum(x[i][p] for i in range(n) for p in range(m)))
        
        # Solver with optimized parameters
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        solver.parameters.num_search_workers = 4
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(i, p) for i in range(n) for p in range(m) if solver.Value(x[i][p]) == 1]
        else:
            return []