import numpy as np
import numba
from ortools.sat.python import cp_model

@numba.njit(cache=True)
def _tsp_solve(dist):
    n = dist.shape[0]
    N = 1 << n
    INF = 10**15
    dp = np.full((N, n), INF, dtype=np.int64)
    par = np.full((N, n), -1, dtype=np.int64)
    # base cases: paths from 0 to each i
    for i in range(1, n):
        m = (1 << 0) | (1 << i)
        dp[m, i] = dist[0, i]
        par[m, i] = 0
    # DP transitions
    for mask in range(N):
        if (mask & 1) == 0 or mask == 1:
            continue
        for j in range(1, n):
            if (mask & (1 << j)) == 0:
                continue
            pm = mask ^ (1 << j)
            if (pm & 1) == 0:
                continue
            best = INF
            bprev = -1
            for k in range(1, n):
                if (pm & (1 << k)) == 0:
                    continue
                cost = dp[pm, k] + dist[k, j]
                if cost < best:
                    best = cost
                    bprev = k
            dp[mask, j] = best
            par[mask, j] = bprev
    # close tour
    full = (1 << n) - 1
    best_cost = INF
    last = 1
    for j in range(1, n):
        cost = dp[full, j] + dist[j, 0]
        if cost < best_cost:
            best_cost = cost
            last = j
    # reconstruct path
    path = np.empty(n + 1, dtype=np.int64)
    path[0] = 0
    path[n] = 0
    mask = full
    cur = last
    for idx in range(n - 1, 0, -1):
        path[idx] = cur
        prev = par[mask, cur]
        mask ^= (1 << cur)
        cur = prev
    return path

# warm up JIT
_dummy = np.zeros((1, 1), dtype=np.int64)
_tsp_solve(_dummy)

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n <= 1:
            return [0, 0]
        # exact Held-Karp for small n
        if n <= 13:
            dist = np.array(problem, dtype=np.int64)
            sol = _tsp_solve(dist)
            return sol.tolist()
        # CP-SAT fallback for larger n
        model = cp_model.CpModel()
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = model.NewBoolVar(f"x[{i},{j}]")
        model.AddCircuit([(u, v, x[u, v]) for (u, v) in x])
        model.Minimize(sum(problem[u][v] * x[u, v] for (u, v) in x))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = kwargs.get("max_time", 60.0)
        solver.parameters.num_search_workers = kwargs.get("num_workers", 8)
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            tour = [0]
            cur = 0
            for _ in range(n - 1):
                for j in range(n):
                    if j != cur and solver.Value(x[cur, j]) == 1:
                        tour.append(j)
                        cur = j
                        break
            tour.append(0)
            return tour
        # fallback trivial
        return list(range(n)) + [0]