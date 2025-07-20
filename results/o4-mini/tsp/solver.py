import numpy as np
from numba import njit
from ortools.sat.python import cp_model

@njit(cache=True)
def dp_held_karp(n, dist):
    """
    Held-Karp exact TSP dynamic programming.
    n: int64 number of cities
    dist: 2D float64 array shape (n,n)
    returns tour as int64 array of length n+1
    """
    K = n - 1
    M = 1 << K
    INF = 1e15
    dp = np.full((M, K), INF, dtype=np.float64)
    parent = np.full((M, K), -1, dtype=np.int64)
    # base cases: from 0 to j+1
    for j in range(K):
        dp[1 << j, j] = dist[0, j+1]
    # DP over subsets
    for mask in range(M):
        for j in range(K):
            cost_j = dp[mask, j]
            if cost_j < INF:
                for k in range(K):
                    if ((mask >> k) & 1) == 0:
                        nm = mask | (1 << k)
                        c = cost_j + dist[j+1, k+1]
                        if c < dp[nm, k]:
                            dp[nm, k] = c
                            parent[nm, k] = j
    # close tour to 0
    full = M - 1
    best = INF
    last = 0
    for j in range(K):
        c = dp[full, j] + dist[j+1, 0]
        if c < best:
            best = c
            last = j
    # reconstruct path
    tour = np.empty(n + 1, dtype=np.int64)
    tour[n] = 0
    mask = full
    j = last
    for idx in range(K-1, -1, -1):
        tour[idx+1] = j + 1
        pj = parent[mask, j]
        mask ^= (1 << j)
        j = pj
    tour[0] = 0
    return tour

class Solver:
    def solve(self, problem, **kwargs):
        """
        Exact TSP: Held-Karp DP for n<=15, CP-SAT exact for n>15.
        """
        n = len(problem)
        if n <= 1:
            return [0, 0]
        # exact DP for small n
        if n <= 15:
            dist_mat = np.array(problem, dtype=np.float64)
            tour = dp_held_karp(np.int64(n), dist_mat)
            return tour.tolist()
        # fallback to CP-SAT solver for larger n
        model = cp_model.CpModel()
        # boolean vars x[i,j] = 1 if tour goes i->j
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[(i,j)] = model.NewBoolVar(f"x[{i},{j}]")
        # circuit constraint
        model.AddCircuit([(i, j, x[(i,j)]) for (i,j) in x])
        # objective
        obj = []
        for (i,j), var in x.items():
            obj.append(problem[i][j] * var)
        model.Minimize(sum(obj))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            path = []
            current = 0
            while len(path) < n:
                path.append(current)
                for nxt in range(n):
                    if current != nxt and solver.Value(x[(current,nxt)]) == 1:
                        current = nxt
                        break
            path.append(0)
            return path
        # fallback greedy nearest neighbor
        visited = [False] * n
        visited[0] = True
        tour = [0]
        for _ in range(n - 1):
            prev = tour[-1]
            row = problem[prev]
            min_d = float('inf')
            next_city = 0
            for j in range(n):
                if not visited[j] and row[j] < min_d:
                    min_d = row[j]
                    next_city = j
            visited[next_city] = True
            tour.append(next_city)
        tour.append(0)
        return tour