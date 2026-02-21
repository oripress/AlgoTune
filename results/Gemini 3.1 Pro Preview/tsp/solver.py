import numpy as np
from numba import njit
from typing import Any, List
from ortools.sat.python import cp_model

@njit
def held_karp(dist):
    n = dist.shape[0]
    
    dp = np.full((1 << (n - 1), n), 1000000000, dtype=np.int32)
    parent = np.zeros((1 << (n - 1), n), dtype=np.int32)
    
    for i in range(1, n):
        dp[1 << (i - 1)][i] = dist[0][i]
        parent[1 << (i - 1)][i] = 0
        
    for mask in range(1, 1 << (n - 1)):
        if (mask & (mask - 1)) == 0:
            continue
            
        for i in range(1, n):
            if (mask & (1 << (i - 1))):
                prev_mask = mask ^ (1 << (i - 1))
                for j in range(1, n):
                    if (prev_mask & (1 << (j - 1))):
                        cost = dp[prev_mask][j] + dist[j][i]
                        if cost < dp[mask][i]:
                            dp[mask][i] = cost
                            parent[mask][i] = j
                            
    min_cost = 1000000000
    last_node = -1
    full_mask = (1 << (n - 1)) - 1
    for i in range(1, n):
        cost = dp[full_mask][i] + dist[i][0]
        if cost < min_cost:
            min_cost = cost
            last_node = i
            
    res = np.zeros(n + 1, dtype=np.int32)
    res[0] = 0
    res[n] = 0
    
    curr_mask = full_mask
    curr_node = last_node
    
    for i in range(n - 1, 0, -1):
        res[i] = curr_node
        next_node = parent[curr_mask][curr_node]
        curr_mask ^= (1 << (curr_node - 1))
        curr_node = next_node
        
    return res

class Solver:
    def __init__(self):
        dummy = np.array([[0, 1], [1, 0]], dtype=np.int32)
        held_karp(dummy)

    def solve(self, problem: List[List[int]], **kwargs) -> Any:
        n = len(problem)
        if n <= 1:
            return [0, 0]
            
        if n <= 20:
            dist = np.array(problem, dtype=np.int32)
            res = held_karp(dist)
            return res.tolist()

        model = cp_model.CpModel()

        x = {}
        arcs = []
        vars_list = []
        coeffs_list = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    var = model.NewBoolVar("")
                    x[i, j] = var
                    arcs.append((i, j, var))
                    vars_list.append(var)
                    coeffs_list.append(problem[i][j])

        model.AddCircuit(arcs)
        model.Minimize(cp_model.LinearExpr.WeightedSum(vars_list, coeffs_list))

        solver = cp_model.CpSolver()
        solver.parameters.cp_model_presolve = False
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            path = []
            current_city = 0
            while len(path) < n:
                path.append(current_city)
                for next_city in range(n):
                    if current_city != next_city and solver.BooleanValue(x[current_city, next_city]):
                        current_city = next_city
                        break
            path.append(0)
            return path
        else:
            return []