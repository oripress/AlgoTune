from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n <= 1:
            return [0, 0]
        if n <= 15:
            return self.held_karp(problem)
        else:
            return self.cp_sat_solver(problem)
    
    def held_karp(self, problem):
        n = len(problem)
        total_masks = 1 << n
        dp = [[10**18] * n for _ in range(total_masks)]
        parent = [[-1] * n for _ in range(total_masks)]
        dp[1][0] = 0
        
        for mask in range(total_masks):
            for i in range(n):
                if dp[mask][i] == 10**18:
                    continue
                if not (mask & (1 << i)):
                    continue
                for j in range(n):
                    if mask & (1 << j):
                        continue
                    new_mask = mask | (1 << j)
                    new_cost = dp[mask][i] + problem[i][j]
                    if new_cost < dp[new_mask][j]:
                        dp[new_mask][j] = new_cost
                        parent[new_mask][j] = i
        
        full_mask = total_masks - 1
        best_cost = 10**18
        best_j = -1
        for j in range(1, n):
            total_cost = dp[full_mask][j] + problem[j][0]
            if total_cost < best_cost:
                best_cost = total_cost
                best_j = j
        
        if best_j == -1:
            return [0, 0]
        
        current = best_j
        mask = full_mask
        path = []
        while current != 0:
            path.append(current)
            prev = parent[mask][current]
            mask = mask ^ (1 << current)
            current = prev
        
        path.reverse()
        tour = [0] + path + [0]
        return tour
    
    def cp_sat_solver(self, problem):
        n = len(problem)
        model = cp_model.CpModel()
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = model.NewBoolVar(f"x[{i},{j}]")
        
        arcs = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    arcs.append((i, j, x[i, j]))
        model.AddCircuit(arcs)
        
        objective_expr = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    objective_expr.append(problem[i][j] * x[i, j])
        model.Minimize(sum(objective_expr))
        
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            path = []
            current_city = 0
            while len(path) < n:
                path.append(current_city)
                for next_city in range(n):
                    if current_city == next_city:
                        continue
                    if solver.Value(x[current_city, next_city]) == 1:
                        current_city = next_city
                        break
            path.append(0)
            return path
        else:
            return []