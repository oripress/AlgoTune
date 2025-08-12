from __future__ import annotations
from typing import List

# MaxSAT solver from python‑sat
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> List[int]:
        """
        Solves the maximum independent set problem using OR‑Tools CP‑SAT.
        Returns a list of vertex indices forming a maximum independent set.
        """
        # Import locally to avoid unnecessary overhead if the method is not used.
        from ortools.sat.python import cp_model

        n = len(problem)
        if n == 0:
            return []

        model = cp_model.CpModel()
        # Boolean variable for each vertex: 1 if selected.
        vars = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Independence constraints: at most one endpoint of each edge can be selected.
        for i in range(n):
            row = problem[i]
            for j in range(i + 1, n):
                if row[j] == 1:
                    model.Add(vars[i] + vars[j] <= 1)

        # Maximize the number of selected vertices.
        model.Maximize(sum(vars))

        solver = cp_model.CpSolver()
        # Enable parallel search and keep a modest time limit.
        solver.parameters.max_time_in_seconds = 5.0
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(vars[i]) == 1]
        else:
            # Fallback: return empty set (should not happen for feasible instances).
            return []