from typing import Any, List

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> Any:
        """
        Solve the Traveling Salesman Problem.
        For modest sizes (n ≤ 15) we use an exact
        Held‑Karp dynamic‑programming algorithm which is
        O(n²·2ⁿ) and very fast in pure Python.
        For larger instances we fall back to OR‑Tools
        CP‑SAT with a short time budget.
        """
        n = len(problem)
        if n <= 1:
            return [0, 0]

        # Exact DP for small n
        if n <= 15:
            INF = 10**12
            # dp[mask][i] = min cost to reach i with visited set mask
            dp = [[INF] * n for _ in range(1 << n)]
            # parent[mask][i] = previous city before i in optimal path for mask
            parent = [[-1] * n for _ in range(1 << n)]

            start_mask = 1 << 0
            dp[start_mask][0] = 0

            for mask in range(1 << n):
                if not (mask & start_mask):
                    continue  # must include start city
                for i in range(n):
                    if not (mask & (1 << i)):
                        continue
                    cur_cost = dp[mask][i]
                    if cur_cost == INF:
                        continue
                    # try to go to a new city j
                    for j in range(n):
                        if mask & (1 << j):
                            continue
                        new_mask = mask | (1 << j)
                        new_cost = cur_cost + problem[i][j]
                        if new_cost < dp[new_mask][j]:
                            dp[new_mask][j] = new_cost
                            parent[new_mask][j] = i

            full_mask = (1 << n) - 1
            # close the tour by returning to 0
            best_cost = INF
            last_city = -1
            for i in range(1, n):
                tour_cost = dp[full_mask][i] + problem[i][0]
                if tour_cost < best_cost:
                    best_cost = tour_cost
                    last_city = i

            # reconstruct path
            tour = [0]
            mask = full_mask
            city = last_city
            while city != -1:
                tour.append(city)
                prev_city = parent[mask][city]
                mask ^= (1 << city)
                city = prev_city
            tour.append(0)  # return to start
            tour.reverse()  # we built it backwards
            return tour

        # Fallback for larger instances: quick CP‑SAT search
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = model.NewBoolVar(f"x_{i}_{j}")

        arcs = [(i, j, x[i, j]) for i in range(n) for j in range(n) if i != j]
        model.AddCircuit(arcs)

        objective = sum(problem[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)
        model.Minimize(objective)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 2.0   # short budget
        solver.parameters.num_search_workers = 4
        solver.parameters.log_search_progress = False

        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            tour = []
            current = 0
            while len(tour) < n:
                tour.append(current)
                for nxt in range(n):
                    if current != nxt and solver.Value(x[current, nxt]) == 1:
                        current = nxt
                        break
            tour.append(0)
            return tour

        # As a last resort, return a simple greedy tour
        visited = [False] * n
        visited[0] = True
        tour = [0]
        current = 0
        for _ in range(n - 1):
            next_city = min(
                (j for j in range(n) if not visited[j]),
                key=lambda j: problem[current][j],
            )
            visited[next_city] = True
            tour.append(next_city)
            current = next_city
        tour.append(0)
        return tour