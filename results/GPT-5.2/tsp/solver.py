from __future__ import annotations

from typing import Any, List

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None  # type: ignore[assignment]

def _held_karp_path_py(dist: np.ndarray) -> np.ndarray:
    """Exact Held–Karp DP (pure Python/numpy fallback)."""
    n = int(dist.shape[0])
    if n <= 1:
        return np.array([0, 0], dtype=np.int64)
    m = n - 1
    size = 1 << m
    inf = np.int64(1 << 60)

    dp = np.full((size, m), inf, dtype=np.int64)
    parent = np.full((size, m), -1, dtype=np.int16)

    for j in range(m):
        mask = 1 << j
        dp[mask, j] = dist[0, j + 1]

    for mask in range(1, size):
        if (mask & (mask - 1)) == 0:
            continue
        for j in range(m):
            if (mask & (1 << j)) == 0:
                continue
            prev = mask ^ (1 << j)
            best = inf
            bestk = -1
            for k in range(m):
                if (prev & (1 << k)) == 0:
                    continue
                cand = dp[prev, k] + dist[k + 1, j + 1]
                if cand < best:
                    best = cand
                    bestk = k
            dp[mask, j] = best
            parent[mask, j] = np.int16(bestk)

    full = size - 1
    best_cost = inf
    best_end = 0
    for j in range(m):
        cand = dp[full, j] + dist[j + 1, 0]
        if cand < best_cost:
            best_cost = cand
            best_end = j

    tour = np.empty(n + 1, dtype=np.int64)
    tour[0] = 0
    mask = full
    end = best_end
    for pos in range(n - 1, 0, -1):
        tour[pos] = end + 1
        prev_end = int(parent[mask, end])
        mask ^= 1 << end
        end = prev_end
    tour[n] = 0
    return tour

if njit is not None:

    @njit(cache=True, fastmath=False)
    def _held_karp_path(dist: np.ndarray) -> np.ndarray:
        n = dist.shape[0]
        if n <= 1:
            return np.array([0, 0], dtype=np.int64)

        m = n - 1  # cities 1..n-1
        size = 1 << m
        inf = np.int64(1 << 60)

        dp = np.full((size, m), inf, dtype=np.int64)
        parent = np.full((size, m), -1, dtype=np.int16)

        for j in range(m):
            mask = 1 << j
            dp[mask, j] = dist[0, j + 1]

        for mask in range(1, size):
            if (mask & (mask - 1)) == 0:
                continue
            for j in range(m):
                if (mask & (1 << j)) == 0:
                    continue
                prev = mask ^ (1 << j)
                best = inf
                bestk = -1
                for k in range(m):
                    if (prev & (1 << k)) == 0:
                        continue
                    cand = dp[prev, k] + dist[k + 1, j + 1]
                    if cand < best:
                        best = cand
                        bestk = k
                dp[mask, j] = best
                parent[mask, j] = np.int16(bestk)

        full = size - 1
        best_cost = inf
        best_end = 0
        for j in range(m):
            cand = dp[full, j] + dist[j + 1, 0]
            if cand < best_cost:
                best_cost = cand
                best_end = j

        tour = np.empty(n + 1, dtype=np.int64)
        tour[0] = 0
        mask = full
        end = best_end
        for pos in range(n - 1, 0, -1):
            tour[pos] = end + 1
            prev_end = int(parent[mask, end])
            mask ^= 1 << end
            end = prev_end
        tour[n] = 0
        return tour

else:
    _held_karp_path = _held_karp_path_py  # type: ignore[misc,assignment]

def _greedy_tour(dist: np.ndarray) -> List[int]:
    """Cheap nearest-neighbor tour for CP-SAT hinting."""
    n = int(dist.shape[0])
    unvis = np.ones(n, dtype=np.bool_)
    unvis[0] = False
    tour = [0]
    cur = 0
    for _ in range(n - 1):
        best = -1
        bestd = 1 << 62
        row = dist[cur]
        for j in range(n):
            if unvis[j]:
                d = int(row[j])
                if d < bestd:
                    bestd = d
                    best = j
        cur = best
        unvis[cur] = False
        tour.append(cur)
    tour.append(0)
    return tour

class Solver:
    def __init__(self, hk_max_n: int = 20) -> None:
        self.hk_max_n = hk_max_n
        self._compiled = False

        # Compile numba kernel in init (not counted in solve runtime).
        if njit is not None:
            try:
                dummy = np.array(
                    [
                        [0, 1, 2, 3],
                        [1, 0, 4, 5],
                        [2, 4, 0, 6],
                        [3, 5, 6, 0],
                    ],
                    dtype=np.int64,
                )
                _held_karp_path(dummy)
                self._compiled = True
            except Exception:
                # If compilation fails for any reason, we'll still work via CP-SAT.
                self._compiled = False

    def solve(self, problem: list[list[int]], **kwargs: Any) -> List[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0]

        dist = np.asarray(problem, dtype=np.int64)

        # Fast exact DP for small instances.
        if n <= self.hk_max_n:
            return _held_karp_path(dist).tolist()

        # Fallback: exact CP-SAT (disable logging, add hint).
        from ortools.sat.python import cp_model  # type: ignore

        model = cp_model.CpModel()
        x: dict[tuple[int, int], Any] = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[(i, j)] = model.NewBoolVar(f"x[{i},{j}]")

        model.AddCircuit([(i, j, var) for (i, j), var in x.items()])
        model.Minimize(sum(int(dist[i, j]) * x[(i, j)] for (i, j) in x.keys()))

        # Hint with a fast greedy tour.
        tour = _greedy_tour(dist)
        for a, b in zip(tour[:-1], tour[1:]):
            if a != b:
                model.AddHint(x[(a, b)], 1)

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        solver.parameters.cp_model_presolve = True
        solver.parameters.linearization_level = 2
        try:
            import os

            solver.parameters.num_search_workers = min(8, os.cpu_count() or 1)
        except Exception:
            pass

        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return []

        nxt = [-1] * n
        for i in range(n):
            for j in range(n):
                if i != j and solver.Value(x[(i, j)]) == 1:
                    nxt[i] = j
                    break

        out = [0]
        cur = 0
        for _ in range(n - 1):
            cur = nxt[cur]
            out.append(cur)
        out.append(0)
        return out