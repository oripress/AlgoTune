from typing import Any

import numpy as np
from numba import njit
from ortools.sat.python import cp_model

@njit(cache=True)
def _held_karp_tsp(dist: np.ndarray, inf: np.int64) -> np.ndarray:
    n = dist.shape[0]

    if n <= 1:
        out = np.empty(2, dtype=np.int64)
        out[0] = 0
        out[1] = 0
        return out

    if n == 2:
        out = np.empty(3, dtype=np.int64)
        out[0] = 0
        out[1] = 1
        out[2] = 0
        return out

    m = n - 1
    size = 1 << m
    total = size * n

    dp = np.empty(total, dtype=np.int64)
    parent = np.empty(total, dtype=np.int32)

    for idx in range(total):
        dp[idx] = inf
        parent[idx] = -1

    for j in range(1, n):
        mask = 1 << (j - 1)
        idx = mask * n + j
        dp[idx] = dist[0, j]
        parent[idx] = 0

    for mask in range(1, size):
        if (mask & (mask - 1)) == 0:
            continue

        mask_base = mask * n
        for j in range(1, n):
            bit = 1 << (j - 1)
            if (mask & bit) == 0:
                continue

            prev_mask = mask ^ bit
            prev_base = prev_mask * n
            best = inf
            best_k = -1

            for k in range(1, n):
                if (prev_mask & (1 << (k - 1))) == 0:
                    continue

                cand = dp[prev_base + k] + dist[k, j]
                if cand < best:
                    best = cand
                    best_k = k

            dp[mask_base + j] = best
            parent[mask_base + j] = best_k

    full_mask = size - 1
    full_base = full_mask * n
    best_cost = inf
    last = -1

    for j in range(1, n):
        cand = dp[full_base + j] + dist[j, 0]
        if cand < best_cost:
            best_cost = cand
            last = j

    path = np.empty(n + 1, dtype=np.int64)
    path[0] = 0
    path[n] = 0

    mask = full_mask
    curr = last
    for pos in range(n - 1, 0, -1):
        path[pos] = curr
        prev = parent[mask * n + curr]
        mask ^= 1 << (curr - 1)
        curr = prev

    return path

def _cp_sat_tsp(problem: list[list[int]]) -> list[int]:
    n = len(problem)

    if n <= 1:
        return [0, 0]

    model = cp_model.CpModel()
    x = {
        (i, j): model.NewBoolVar(f"x[{i},{j}]")
        for i in range(n)
        for j in range(n)
        if i != j
    }

    model.AddCircuit([(i, j, var) for (i, j), var in x.items()])
    model.Minimize(
        sum(problem[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)
    )

    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = False
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return []

    path = [0]
    current = 0
    while len(path) < n:
        for nxt in range(n):
            if current != nxt and solver.Value(x[current, nxt]) == 1:
                path.append(nxt)
                current = nxt
                break
    path.append(0)
    return path

class Solver:
    def __init__(self) -> None:
        dummy = np.array([[0, 1], [1, 0]], dtype=np.int64)
        _held_karp_tsp(dummy, np.int64(3))

    def solve(self, problem, **kwargs) -> Any:
        n = len(problem)
        if n <= 1:
            return [0, 0]

        if n <= 21:
            try:
                dist = np.asarray(problem, dtype=np.int64)
                if dist.shape != (n, n):
                    dist = np.array(problem, dtype=np.int64, copy=True)

                max_edge = int(dist.max())
                max_total = np.iinfo(np.int64).max // max(n, 1) - 1
                if max_edge <= max_total:
                    inf = np.int64(max_edge * n + 1)
                    return _held_karp_tsp(dist, inf).tolist()
            except (MemoryError, ValueError, OverflowError):
                pass

        return _cp_sat_tsp(problem)