from typing import Any, List
import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> Any:
        n = len(problem)
        if n == 0:
            return []
        
        # Convert to numpy for faster edge finding
        adj = np.array(problem, dtype=np.int8)
        
        # Find all edges using upper triangle
        rows, cols = np.where(np.triu(adj, k=1) == 1)
        
        if len(rows) == 0:
            return []
        
        model = cp_model.CpModel()
        
        # Create binary variables
        x = [model.NewBoolVar(f'x{i}') for i in range(n)]
        
        # Edge constraints: at least one endpoint must be in cover
        for i in range(len(rows)):
            model.AddBoolOr([x[rows[i]], x[cols[i]]])
        
        # Objective: minimize number of vertices in cover
        model.Minimize(sum(x))
        
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 4
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(x[i]) == 1]
        else:
            return list(range(n))