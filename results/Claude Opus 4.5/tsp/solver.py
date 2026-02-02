import numpy as np
from numba import jit
from ortools.sat.python import cp_model

@jit(nopython=True, cache=True)
def held_karp_dp(dist, n):
    INF = 10**15
    
    num_masks = 1 << n
    dp = np.full((num_masks, n), INF, dtype=np.int64)
    parent = np.full((num_masks, n), -1, dtype=np.int32)
    
    dp[1, 0] = 0
    
    for mask in range(1, num_masks):
        if not (mask & 1):
            continue
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            cur_cost = dp[mask, last]
            if cur_cost >= INF:
                continue
            for next_city in range(n):
                if mask & (1 << next_city):
                    continue
                new_mask = mask | (1 << next_city)
                new_cost = cur_cost + dist[last, next_city]
                if new_cost < dp[new_mask, next_city]:
                    dp[new_mask, next_city] = new_cost
                    parent[new_mask, next_city] = last
    
    full_mask = num_masks - 1
    best_cost = INF
    best_last = 0
    for last in range(1, n):
        cost = dp[full_mask, last] + dist[last, 0]
        if cost < best_cost:
            best_cost = cost
            best_last = last
    
    return parent, best_last

@jit(nopython=True, cache=True)
def reconstruct_path(parent, best_last, n):
    full_mask = (1 << n) - 1
    path = np.zeros(n + 1, dtype=np.int32)
    
    mask = full_mask
    current = best_last
    idx = n - 1
    while current != 0:
        path[idx] = current
        prev = parent[mask, current]
        mask ^= (1 << current)
        current = prev
        idx -= 1
    path[0] = 0
    path[n] = 0
    
    return path

class Solver:
    def __init__(self):
        # Warm up numba compilation
        dist = np.array([[0, 1], [1, 0]], dtype=np.int64)
        held_karp_dp(dist, 2)
        parent = np.array([[0, 0], [0, 0]], dtype=np.int32)
        reconstruct_path(parent, 1, 2)
    
    def solve(self, problem, **kwargs):
        n = len(problem)
        
        if n <= 1:
            return [0, 0]
        
        if n == 2:
            return [0, 1, 0]
        
        # For small-medium instances, use Held-Karp DP with numba
        if n <= 20:
            return self._held_karp_numba(problem, n)
        
        # For larger instances, use CP-SAT solver
        return self._ortools_cpsat(problem, n)
    
    def _held_karp_numba(self, problem, n):
        dist = np.array(problem, dtype=np.int64)
        parent, best_last = held_karp_dp(dist, n)
        path = reconstruct_path(parent, best_last, n)
        return path.tolist()
    
    def _ortools_cpsat(self, problem, n):
        model = cp_model.CpModel()
        x = {(i, j): model.NewBoolVar(f"x[{i},{j}]") for i in range(n) for j in range(n) if i != j}
        model.AddCircuit([(u, v, var) for (u, v), var in x.items()])
        model.Minimize(sum(problem[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j))
        
        solver = cp_model.CpSolver()
        solver.parameters.num_workers = 8
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            path = []
            current_city = 0
            while len(path) < n:
                path.append(current_city)
                for next_city in range(n):
                    if current_city != next_city and solver.Value(x[current_city, next_city]) == 1:
                        current_city = next_city
                        break
            path.append(0)
            return path
        else:
            return []