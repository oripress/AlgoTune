from typing import Any, Iterable, Sequence, Tuple, List


class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Multi-Dimensional Knapsack Problem solver.

        Returns list of selected item indices. Empty list on failure.

        Input formats supported:
        - An object with attributes: value, demand, supply
        - A tuple/list: (value, demand, supply)
        """
        # Parse input
        try:
            if hasattr(problem, "value") and hasattr(problem, "demand") and hasattr(
                problem, "supply"
            ):
                value = list(problem.value)
                demand = [list(row) for row in problem.demand]
                supply = list(problem.supply)
            else:
                value, demand, supply = problem  # type: ignore[misc]
                value = list(value)
                demand = [list(row) for row in demand]
                supply = list(supply)
        except Exception:
            return []

        n = len(value)
        if n == 0:
            return []
        k = len(supply)
        if any(len(row) != k for row in demand) or len(demand) != n:
            return []

        # Import CP-SAT
        try:
            from ortools.sat.python import cp_model
        except Exception:
            return []

        # Build model exactly as reference for identical outputs
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Constraints
        for r in range(k):
            model.Add(
                sum(int(demand[i][r]) * x[i] for i in range(n)) <= int(supply[r])
            )

        # Objective
        model.Maximize(sum(int(value[i]) * x[i] for i in range(n)))

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x[i])]
        return []