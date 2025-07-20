from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        # Unpack tuple/list input: (values, demands, supply)
        try:
            values, demands, supply = problem
            values = list(values)
            demands = [list(d) for d in demands]
            supply = list(supply)
        except Exception:
            return []
        n = len(values)
        k = len(supply)
        if n == 0 or k == 0:
            return []
        # Build CP-SAT model
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        # Resource constraints
        for j in range(k):
            model.Add(sum(x[i] * demands[i][j] for i in range(n)) <= supply[j])
        # Objective
        model.Maximize(sum(x[i] * values[i] for i in range(n)))
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x[i])]
        return []

# Module-level entrypoint
solver = Solver()
def solve(problem, **kwargs):
    return solver.solve(problem, **kwargs)