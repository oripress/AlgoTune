from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        """
        Exact solver for the multi-dimensional knapsack problem using OR-Tools CP-SAT.
        Returns list of item indices maximizing value under resource constraints.
        """
        # Extract data
        try:
            v = problem.value
            d = problem.demand
            s = problem.supply
        except Exception:
            try:
                v, d, s = problem
            except Exception:
                return []
        n = len(v)
        if n == 0:
            return []
        k = len(s)
        # Build CP-SAT model
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        # Resource constraints
        for r in range(k):
            model.Add(sum(x[i] * d[i][r] for i in range(n)) <= s[r])
        # Objective: maximize total value
        model.Maximize(sum(x[i] * v[i] for i in range(n)))
        # Greedy initial solution hint
        try:
            ratios = []
            for i in range(n):
                weight = 0.0
                for j in range(k):
                    if s[j] > 0:
                        weight += d[i][j] / float(s[j])
                    elif d[i][j] > 0:
                        weight = float('inf')
                        break
                ratio = v[i] / weight if weight > 0 else float('inf')
                ratios.append((ratio, i))
            ratios.sort(reverse=True)
            supply_rem = list(s)
            for _, idx in ratios:
                if all(d[idx][j] <= supply_rem[j] for j in range(k)):
                    for j in range(k):
                        supply_rem[j] -= d[idx][j]
                    model.AddHint(x[idx], 1)
        except Exception:
            pass
        # Solve
        solver = cp_model.CpSolver()
        # Parallel search
        solver.parameters.num_search_workers = kwargs.get("num_search_workers", 8)
        status = solver.Solve(model)
        # Extract solution if found
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x[i])]
        # Return empty if no feasible solution
        return []