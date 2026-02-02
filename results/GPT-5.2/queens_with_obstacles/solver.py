from __future__ import annotations

import os
from collections import deque
from typing import Any, Callable

import numpy as np
from ortools.sat.python import cp_model

# Optional Numba acceleration for segment labeling (compile in __init__).
try:
    from numba import njit  # type: ignore

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False
    njit = None  # type: ignore

# ----------------------------
# Segment labeling (obstacle-aware)
# ----------------------------
def _label_row_segments_py(obs: np.ndarray) -> tuple[np.ndarray, int]:
    n, m = obs.shape
    seg_id = np.full((n, m), -1, dtype=np.int32)
    seg = 0
    for r in range(n):
        c = 0
        while c < m:
            if obs[r, c]:
                c += 1
                continue
            while c < m and not obs[r, c]:
                seg_id[r, c] = seg
                c += 1
            seg += 1
    return seg_id, seg

def _label_col_segments_py(obs: np.ndarray) -> tuple[np.ndarray, int]:
    n, m = obs.shape
    seg_id = np.full((n, m), -1, dtype=np.int32)
    seg = 0
    for c in range(m):
        r = 0
        while r < n:
            if obs[r, c]:
                r += 1
                continue
            while r < n and not obs[r, c]:
                seg_id[r, c] = seg
                r += 1
            seg += 1
    return seg_id, seg

def _label_diag_segments_py(obs: np.ndarray) -> tuple[np.ndarray, int]:
    """Main diagonals (r+1,c+1), split by obstacles."""
    n, m = obs.shape
    seg_id = np.full((n, m), -1, dtype=np.int32)
    seg = 0

    for c0 in range(m):  # start points on top row
        r, c = 0, c0
        while r < n and c < m:
            if obs[r, c]:
                r += 1
                c += 1
                continue
            while r < n and c < m and not obs[r, c]:
                seg_id[r, c] = seg
                r += 1
                c += 1
            seg += 1

    for r0 in range(1, n):  # start points on left col (excluding (0,0))
        r, c = r0, 0
        while r < n and c < m:
            if obs[r, c]:
                r += 1
                c += 1
                continue
            while r < n and c < m and not obs[r, c]:
                seg_id[r, c] = seg
                r += 1
                c += 1
            seg += 1

    return seg_id, seg

def _label_anti_diag_segments_py(obs: np.ndarray) -> tuple[np.ndarray, int]:
    """Anti-diagonals (r+1,c-1), split by obstacles."""
    n, m = obs.shape
    seg_id = np.full((n, m), -1, dtype=np.int32)
    seg = 0

    for c0 in range(m):  # start points on top row
        r, c = 0, c0
        while r < n and c >= 0:
            if obs[r, c]:
                r += 1
                c -= 1
                continue
            while r < n and c >= 0 and not obs[r, c]:
                seg_id[r, c] = seg
                r += 1
                c -= 1
            seg += 1

    for r0 in range(1, n):  # start points on right col (excluding (0,m-1))
        r, c = r0, m - 1
        while r < n and c >= 0:
            if obs[r, c]:
                r += 1
                c -= 1
                continue
            while r < n and c >= 0 and not obs[r, c]:
                seg_id[r, c] = seg
                r += 1
                c -= 1
            seg += 1

    return seg_id, seg

if _HAS_NUMBA:

    @njit(cache=False)
    def _label_row_segments_nb(obs: np.ndarray) -> tuple[np.ndarray, int]:
        n, m = obs.shape
        seg_id = np.empty((n, m), dtype=np.int32)
        seg_id[:] = -1
        seg = 0
        for r in range(n):
            c = 0
            while c < m:
                if obs[r, c]:
                    c += 1
                else:
                    while c < m and not obs[r, c]:
                        seg_id[r, c] = seg
                        c += 1
                    seg += 1
        return seg_id, seg

    @njit(cache=False)
    def _label_col_segments_nb(obs: np.ndarray) -> tuple[np.ndarray, int]:
        n, m = obs.shape
        seg_id = np.empty((n, m), dtype=np.int32)
        seg_id[:] = -1
        seg = 0
        for c in range(m):
            r = 0
            while r < n:
                if obs[r, c]:
                    r += 1
                else:
                    while r < n and not obs[r, c]:
                        seg_id[r, c] = seg
                        r += 1
                    seg += 1
        return seg_id, seg

    @njit(cache=False)
    def _label_diag_segments_nb(obs: np.ndarray) -> tuple[np.ndarray, int]:
        n, m = obs.shape
        seg_id = np.empty((n, m), dtype=np.int32)
        seg_id[:] = -1
        seg = 0

        for c0 in range(m):
            r, c = 0, c0
            while r < n and c < m:
                if obs[r, c]:
                    r += 1
                    c += 1
                else:
                    while r < n and c < m and not obs[r, c]:
                        seg_id[r, c] = seg
                        r += 1
                        c += 1
                    seg += 1

        for r0 in range(1, n):
            r, c = r0, 0
            while r < n and c < m:
                if obs[r, c]:
                    r += 1
                    c += 1
                else:
                    while r < n and c < m and not obs[r, c]:
                        seg_id[r, c] = seg
                        r += 1
                        c += 1
                    seg += 1

        return seg_id, seg

    @njit(cache=False)
    def _label_anti_diag_segments_nb(obs: np.ndarray) -> tuple[np.ndarray, int]:
        n, m = obs.shape
        seg_id = np.empty((n, m), dtype=np.int32)
        seg_id[:] = -1
        seg = 0

        for c0 in range(m):
            r, c = 0, c0
            while r < n and c >= 0:
                if obs[r, c]:
                    r += 1
                    c -= 1
                else:
                    while r < n and c >= 0 and not obs[r, c]:
                        seg_id[r, c] = seg
                        r += 1
                        c -= 1
                    seg += 1

        for r0 in range(1, n):
            r, c = r0, m - 1
            while r < n and c >= 0:
                if obs[r, c]:
                    r += 1
                    c -= 1
                else:
                    while r < n and c >= 0 and not obs[r, c]:
                        seg_id[r, c] = seg
                        r += 1
                        c -= 1
                    seg += 1

        return seg_id, seg

    _label_row_segments: Callable[[np.ndarray], tuple[np.ndarray, int]] = _label_row_segments_nb
    _label_col_segments: Callable[[np.ndarray], tuple[np.ndarray, int]] = _label_col_segments_nb
    _label_diag_segments: Callable[[np.ndarray], tuple[np.ndarray, int]] = _label_diag_segments_nb
    _label_anti_diag_segments: Callable[[np.ndarray], tuple[np.ndarray, int]] = (
        _label_anti_diag_segments_nb
    )
else:  # pragma: no cover
    _label_row_segments = _label_row_segments_py
    _label_col_segments = _label_col_segments_py
    _label_diag_segments = _label_diag_segments_py
    _label_anti_diag_segments = _label_anti_diag_segments_py

# ----------------------------
# Upper bound helpers
# ----------------------------
def _hopcroft_karp(adj: list[list[int]], n_left: int, n_right: int) -> int:
    """Maximum bipartite matching size (left: 0..n_left-1)."""
    match_l = [-1] * n_left
    match_r = [-1] * n_right
    dist = [0] * n_left
    q: deque[int] = deque()

    inf = 1 << 30

    def bfs() -> bool:
        q.clear()
        for u in range(n_left):
            if match_l[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = inf
        found = False
        while q:
            u = q.popleft()
            du = dist[u] + 1
            for v in adj[u]:
                mu = match_r[v]
                if mu == -1:
                    found = True
                elif dist[mu] == inf:
                    dist[mu] = du
                    q.append(mu)
        return found

    def dfs(u: int) -> bool:
        for v in adj[u]:
            mu = match_r[v]
            if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                match_l[u] = v
                match_r[v] = u
                return True
        dist[u] = inf
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if match_l[u] == -1 and dfs(u):
                matching += 1
    return matching

# ----------------------------
# Lower bound helper
# ----------------------------
def _greedy_pack(
    row_id: np.ndarray,
    col_id: np.ndarray,
    diag_id: np.ndarray,
    anti_id: np.ndarray,
    order: np.ndarray | None,
    n_row: int,
    n_col: int,
    n_diag: int,
    n_anti: int,
) -> np.ndarray:
    """Returns indices of selected items (greedy)."""
    max_seg = max(n_row, n_col, n_diag, n_anti)
    chosen: list[int] = []

    if order is None:
        order_it = range(int(row_id.shape[0]))
    else:
        order_it = order

    # Bitset is fast for small segment universes; bytearray scales better for large ids.
    if max_seg <= 256:
        used_row = 0
        used_col = 0
        used_d = 0
        used_a = 0
        for i in order_it:
            ii = int(i)
            r = int(row_id[ii])
            c = int(col_id[ii])
            d = int(diag_id[ii])
            a = int(anti_id[ii])
            if ((used_row >> r) & 1) or ((used_col >> c) & 1) or ((used_d >> d) & 1) or (
                (used_a >> a) & 1
            ):
                continue
            used_row |= 1 << r
            used_col |= 1 << c
            used_d |= 1 << d
            used_a |= 1 << a
            chosen.append(ii)
    else:
        ur = bytearray(n_row)
        uc = bytearray(n_col)
        ud = bytearray(n_diag)
        ua = bytearray(n_anti)
        for i in order_it:
            ii = int(i)
            r = int(row_id[ii])
            c = int(col_id[ii])
            d = int(diag_id[ii])
            a = int(anti_id[ii])
            if ur[r] or uc[c] or ud[d] or ua[a]:
                continue
            ur[r] = 1
            uc[c] = 1
            ud[d] = 1
            ua[a] = 1
            chosen.append(ii)

    return np.fromiter(chosen, dtype=np.int32)

# ----------------------------
# Exact fast path for small k: maximum clique (compatibility graph) with bitsets
# ----------------------------
def _max_clique_bitset(adj: list[int], initial_best: list[int]) -> list[int]:
    """
    Maximum clique in graph with bitset adjacency.
    adj[v] is an int bitmask of neighbors.
    """
    n = len(adj)
    all_mask = (1 << n) - 1

    best = initial_best[:]  # list of vertices
    best_len = len(best)

    def lsb_index(x: int) -> int:
        return (x & -x).bit_length() - 1

    # Greedy coloring upper bound for clique search.
    def color_sort(cand: int) -> tuple[list[int], list[int]]:
        order: list[int] = []
        colors: list[int] = []
        cset = cand
        color = 0
        while cset:
            color += 1
            avail = cset
            while avail:
                v = lsb_index(avail)
                vb = 1 << v
                avail &= ~vb
                cset &= ~vb
                # same color must be independent => remove neighbors from avail
                avail &= ~adj[v]
                order.append(v)
                colors.append(color)
        return order, colors

    clique: list[int] = []

    def expand(cand: int) -> None:
        nonlocal best_len, best
        if not cand:
            if len(clique) > best_len:
                best = clique[:]
                best_len = len(clique)
            return

        order, colors = color_sort(cand)
        for i in range(len(order) - 1, -1, -1):
            if len(clique) + colors[i] <= best_len:
                return
            v = order[i]
            vb = 1 << v
            clique.append(v)
            new_cand = cand & adj[v]
            if new_cand:
                expand(new_cand)
            else:
                if len(clique) > best_len:
                    best = clique[:]
                    best_len = len(clique)
            clique.pop()
            cand &= ~vb

    expand(all_mask)
    return best

class Solver:
    def __init__(self) -> None:
        self._default_workers = min(8, os.cpu_count() or 1)

        # Compile Numba labelers in init (not counted in runtime).
        if _HAS_NUMBA:
            dummy = np.zeros((2, 2), dtype=np.bool_)
            _label_row_segments(dummy)
            _label_col_segments(dummy)
            _label_diag_segments(dummy)
            _label_anti_diag_segments(dummy)

    def solve(self, problem: np.ndarray, **kwargs: Any) -> list[tuple[int, int]]:
        obs = np.asarray(problem, dtype=np.bool_)
        if obs.size == 0:
            return []

        free_pos = np.argwhere(~obs).astype(np.int32, copy=False)
        k = int(free_pos.shape[0])
        if k == 0:
            return []

        # Segment labels (obstacle-aware).
        row_seg, n_row = _label_row_segments(obs)
        col_seg, n_col = _label_col_segments(obs)
        diag_seg, n_diag = _label_diag_segments(obs)
        anti_seg, n_anti = _label_anti_diag_segments(obs)

        rr = free_pos[:, 0]
        cc = free_pos[:, 1]
        row_id = row_seg[rr, cc]
        col_id = col_seg[rr, cc]
        diag_id = diag_seg[rr, cc]
        anti_id = anti_seg[rr, cc]

        # Cheap universal upper bound.
        ub0 = min(n_row, n_col, n_diag, n_anti, k)

        # Very cheap greedy in natural order (no sorting).
        best_idx = _greedy_pack(
            row_id, col_id, diag_id, anti_id, None, n_row, n_col, n_diag, n_anti
        )
        best_len = int(best_idx.size)
        if best_len >= ub0:
            return [(int(rr[i]), int(cc[i])) for i in best_idx.tolist()]

        # Strengthen LB a bit (only when needed).
        row_cnt = np.bincount(row_id, minlength=n_row)
        col_cnt = np.bincount(col_id, minlength=n_col)
        diag_cnt = np.bincount(diag_id, minlength=n_diag)
        anti_cnt = np.bincount(anti_id, minlength=n_anti)
        score = row_cnt[row_id] + col_cnt[col_id] + diag_cnt[diag_id] + anti_cnt[anti_id]
        order = np.argsort(score)  # quicksort by default
        cand = _greedy_pack(row_id, col_id, diag_id, anti_id, order, n_row, n_col, n_diag, n_anti)
        if cand.size > best_idx.size:
            best_idx = cand
            best_len = int(cand.size)
            if best_len >= ub0:
                return [(int(rr[i]), int(cc[i])) for i in best_idx.tolist()]

        # Exact fast path for small k (avoid CP-SAT overhead).
        if k <= 120:
            row_mask = [0] * n_row
            col_mask = [0] * n_col
            diag_mask = [0] * n_diag
            anti_mask = [0] * n_anti
            for i in range(k):
                bit = 1 << i
                row_mask[int(row_id[i])] |= bit
                col_mask[int(col_id[i])] |= bit
                diag_mask[int(diag_id[i])] |= bit
                anti_mask[int(anti_id[i])] |= bit

            all_mask = (1 << k) - 1
            adj: list[int] = [0] * k
            for i in range(k):
                conflict = (
                    row_mask[int(row_id[i])]
                    | col_mask[int(col_id[i])]
                    | diag_mask[int(diag_id[i])]
                    | anti_mask[int(anti_id[i])]
                )
                adj[i] = (all_mask & ~conflict) & ~(1 << i)

            clique = _max_clique_bitset(adj, initial_best=[int(x) for x in best_idx.tolist()])
            return [(int(rr[i]), int(cc[i])) for i in clique]

        # Upper bound via rook/bishop matchings (cheap, often certifies optimality).
        rook_adj: list[list[int]] = [[] for _ in range(n_row)]
        bishop_adj: list[list[int]] = [[] for _ in range(n_diag)]
        for i in range(k):
            rook_adj[int(row_id[i])].append(int(col_id[i]))
            bishop_adj[int(diag_id[i])].append(int(anti_id[i]))

        rook_ub = _hopcroft_karp(rook_adj, n_row, n_col)
        bishop_ub = _hopcroft_karp(bishop_adj, n_diag, n_anti)
        ub = min(int(rook_ub), int(bishop_ub), ub0)

        if best_len >= ub:
            return [(int(rr[i]), int(cc[i])) for i in best_idx.tolist()]

        # CP-SAT exact solve (fallback for larger/harder instances).
        model = cp_model.CpModel()
        vars_ = [model.NewBoolVar(f"q{i}") for i in range(k)]

        row_groups: list[list[cp_model.IntVar]] = [[] for _ in range(n_row)]
        col_groups: list[list[cp_model.IntVar]] = [[] for _ in range(n_col)]
        diag_groups: list[list[cp_model.IntVar]] = [[] for _ in range(n_diag)]
        anti_groups: list[list[cp_model.IntVar]] = [[] for _ in range(n_anti)]

        for i, v in enumerate(vars_):
            row_groups[int(row_id[i])].append(v)
            col_groups[int(col_id[i])].append(v)
            diag_groups[int(diag_id[i])].append(v)
            anti_groups[int(anti_id[i])].append(v)

        for groups in (row_groups, col_groups, diag_groups, anti_groups):
            for g in groups:
                if len(g) > 1:
                    model.AddAtMostOne(g)

        tot = cp_model.LinearExpr.Sum(vars_)
        model.Add(tot <= ub)
        model.Add(tot >= best_len)
        model.Maximize(tot)

        # Hint the greedy solution.
        for i in best_idx.tolist():
            model.AddHint(vars_[int(i)], 1)

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        workers = int(kwargs.get("workers") or self._default_workers)
        solver.parameters.num_search_workers = 1 if k < 220 else workers

        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return []

        out: list[tuple[int, int]] = []
        for i, v in enumerate(vars_):
            if solver.BooleanValue(v):
                out.append((int(rr[i]), int(cc[i])))
        return out