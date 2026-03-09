from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit
from pysat.solvers import Solver as SATSolver

@njit(cache=True)
def _solve_btsp_dp_u8(ranks: np.ndarray) -> np.ndarray:
    n = ranks.shape[0]
    if n == 1:
        return np.array([0, 0], dtype=np.int64)
    if n == 2:
        return np.array([0, 1, 0], dtype=np.int64)

    m = n - 1
    total = 1 << m

    bit_to_idx = np.empty(total, dtype=np.int8)
    for j in range(m):
        bit_to_idx[1 << j] = j

    inf = np.uint8(255)
    dp = np.full((total, m), inf, dtype=np.uint8)

    for j in range(m):
        dp[1 << j, j] = ranks[0, j + 1]

    for mask in range(1, total):
        if (mask & (mask - 1)) == 0:
            continue

        rem = mask
        while rem:
            lb = rem & -rem
            j = bit_to_idx[lb]
            prev_mask = mask ^ lb

            best = inf
            prev_bits = prev_mask
            while prev_bits:
                kb = prev_bits & -prev_bits
                k = bit_to_idx[kb]
                cand = dp[prev_mask, k]
                edge = ranks[k + 1, j + 1]
                if edge > cand:
                    cand = edge
                if cand < best:
                    best = cand
                prev_bits ^= kb

            dp[mask, j] = best
            rem ^= lb

    full = total - 1
    best = inf
    end = 0
    for j in range(m):
        cand = dp[full, j]
        edge = ranks[j + 1, 0]
        if edge > cand:
            cand = edge
        if cand < best:
            best = cand
            end = j

    path = np.empty(n + 1, dtype=np.int64)
    path[0] = 0
    path[n] = 0

    idx = n - 1
    mask = full
    cur = end
    while True:
        path[idx] = cur + 1
        idx -= 1
        prev_mask = mask ^ (1 << cur)
        if prev_mask == 0:
            break

        need = dp[mask, cur]
        prev_bits = prev_mask
        chosen = -1
        while prev_bits:
            kb = prev_bits & -prev_bits
            k = bit_to_idx[kb]
            cand = dp[prev_mask, k]
            edge = ranks[k + 1, cur + 1]
            if edge > cand:
                cand = edge
            if cand == need:
                chosen = k
                break
            prev_bits ^= kb

        mask = prev_mask
        cur = chosen

    return path

def _sat_hamiltonian_cycle(allowed: np.ndarray) -> list[int] | None:
    n = int(allowed.shape[0])
    if n == 1:
        return [0, 0]
    if n == 2:
        return [0, 1, 0] if allowed[0, 1] else None

    deg = allowed.sum(axis=1)
    if np.any(deg < 2):
        return None

    seen = np.zeros(n, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        u = stack.pop()
        nbrs = np.flatnonzero(allowed[u] & ~seen)
        if nbrs.size:
            seen[nbrs] = True
            stack.extend(int(v) for v in nbrs)
    if not np.all(seen):
        return None

    size = n - 1

    def var(v: int, pos: int) -> int:
        return (v - 1) * size + pos

    clauses: list[list[int]] = []

    for pos in range(1, n):
        clauses.append([var(v, pos) for v in range(1, n)])
        for a in range(1, n):
            va = var(a, pos)
            for b in range(a + 1, n):
                clauses.append([-va, -var(b, pos)])

    for v in range(1, n):
        clauses.append([var(v, pos) for pos in range(1, n)])
        for p1 in range(1, n):
            vp = var(v, p1)
            for p2 in range(p1 + 1, n):
                clauses.append([-vp, -var(v, p2)])

    for v in range(1, n):
        if not allowed[0, v]:
            clauses.append([-var(v, 1)])
            clauses.append([-var(v, n - 1)])

    for u in range(1, n):
        for v in range(u + 1, n):
            if allowed[u, v]:
                continue
            for pos in range(1, n - 1):
                clauses.append([-var(u, pos), -var(v, pos + 1)])
                clauses.append([-var(v, pos), -var(u, pos + 1)])

    with SATSolver(bootstrap_with=clauses) as solver:
        if not solver.solve():
            return None
        model = solver.get_model()

    truth = set(lit for lit in model if lit > 0)

    tour = [0] * (n + 1)
    tour[0] = 0
    tour[-1] = 0
    for pos in range(1, n):
        for v in range(1, n):
            if var(v, pos) in truth:
                tour[pos] = v
                break
    return tour

def _heuristic_upper_bound(matrix: np.ndarray) -> float:
    n = matrix.shape[0]

    seq = 0.0
    for i in range(n - 1):
        w = float(matrix[i, i + 1])
        if w > seq:
            seq = w
    w = float(matrix[n - 1, 0])
    if w > seq:
        seq = w

    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    cur = 0
    nn = 0.0
    for _ in range(n - 1):
        row = matrix[cur].copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        w = float(matrix[cur, nxt])
        if w > nn:
            nn = w
        visited[nxt] = True
        cur = nxt
    w = float(matrix[cur, 0])
    if w > nn:
        nn = w

    return nn if nn < seq else seq

class Solver:
    def __init__(self) -> None:
        _solve_btsp_dp_u8(np.array([[0, 1], [1, 0]], dtype=np.uint8))

    def solve(self, problem, **kwargs) -> Any:
        n = len(problem)
        if n == 0:
            return []
        if n == 1:
            return [0, 0]
        if n == 2:
            return [0, 1, 0]
        if n == 3:
            return [0, 1, 2, 0]

        matrix = np.asarray(problem, dtype=np.float64)
        upper_tri = matrix[np.triu_indices(n, 1)]
        values = np.unique(upper_tri)

        if values.size == 1:
            return list(range(n)) + [0]

        if n <= 23:
            ranks_u8 = np.searchsorted(values, matrix).astype(np.uint8)
            return _solve_btsp_dp_u8(ranks_u8).tolist()

        ranks = np.searchsorted(values, matrix)

        mat2 = matrix.copy()
        np.fill_diagonal(mat2, np.inf)
        second_smallest = np.partition(mat2, 1, axis=1)[:, 1]
        lo = int(np.searchsorted(values, float(np.max(second_smallest)), side="left"))

        ub = _heuristic_upper_bound(matrix)
        hi = int(np.searchsorted(values, ub, side="left"))
        if hi < lo:
            hi = lo

        best_tour: list[int] | None = None
        while lo <= hi:
            mid = (lo + hi) // 2
            allowed = ranks <= mid
            np.fill_diagonal(allowed, False)
            tour = _sat_hamiltonian_cycle(allowed)
            if tour is None:
                lo = mid + 1
            else:
                best_tour = tour
                hi = mid - 1

        if best_tour is not None:
            return best_tour

        allowed = ranks <= lo
        np.fill_diagonal(allowed, False)
        tour = _sat_hamiltonian_cycle(allowed)
        return tour if tour is not None else []