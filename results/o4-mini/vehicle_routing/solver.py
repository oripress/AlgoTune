from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)

        model = cp_model.CpModel()
        # arc selection vars
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")

        # each non-depot visited once
        for i in range(n):
            if i == depot:
                continue
            model.Add(sum(x[(j, i)] for j in range(n) if j != i) == 1)
            model.Add(sum(x[(i, j)] for j in range(n) if j != i) == 1)

        # depot degree = K
        model.Add(sum(x[(depot, j)] for j in range(n) if j != depot) == K)
        model.Add(sum(x[(i, depot)] for i in range(n) if i != depot) == K)

        # MTZ for subtours
        u = {}
        for i in range(n):
            if i == depot:
                continue
            u[i] = model.NewIntVar(1, n - 1, f"u_{i}")

        for i in range(n):
            if i == depot:
                continue
            for j in range(n):
                if j == depot or i == j:
                    continue
                model.Add(u[i] + 1 <= u[j] + (n - 1) * (1 - x[(i, j)]))

        # objective
        model.Minimize(sum(D[i][j] * x[(i, j)] for (i, j) in x))

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)

        routes = []
        if status == cp_model.OPTIMAL:
            for j in range(n):
                if j != depot and solver.Value(x[(depot, j)]) == 1:
                    route = [depot, j]
                    cur = j
                    while cur != depot:
                        for k in range(n):
                            if cur != k and solver.Value(x[(cur, k)]) == 1:
                                route.append(k)
                                cur = k
                                break
                    routes.append(route)
        return routes