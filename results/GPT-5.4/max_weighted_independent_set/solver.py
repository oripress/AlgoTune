from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit
from ortools.sat.python import cp_model

@njit(cache=True)
def _popcount(x: np.uint64) -> int:
    c = 0
    one = np.uint64(1)
    while x:
        x = x & (x - one)
        c += 1
    return c

@njit(cache=True)
def _bit_index(bit: np.uint64) -> int:
    i = 0
    while bit > 1:
        bit >>= 1
        i += 1
    return i

@njit(cache=True)
def _greedy_lb(cand_mask: np.uint64, compat: np.ndarray, weights: np.ndarray) -> tuple[int, np.uint64]:
    best_total = 0
    best_sel = np.uint64(0)
    one = np.uint64(1)

    for mode in range(2):
        rem = cand_mask
        total = 0
        sel = np.uint64(0)

        while rem:
            tmp = rem
            choice_v = -1
            choice_bit = np.uint64(0)
            choice_a = -1
            choice_b = -1

            while tmp:
                bit = tmp & (~tmp + one)
                tmp ^= bit
                v = _bit_index(bit)
                deg = _popcount(compat[v] & rem)
                w = int(weights[v])
                if mode == 0:
                    a = (w * 1024) // (deg + 1)
                    b = w
                else:
                    a = w
                    b = -deg

                if choice_v == -1 or a > choice_a or (a == choice_a and b > choice_b):
                    choice_v = v
                    choice_bit = bit
                    choice_a = a
                    choice_b = b

            sel |= choice_bit
            total += int(weights[choice_v])
            rem &= compat[choice_v]

        if total > best_total:
            best_total = total
            best_sel = sel

    return best_total, best_sel

@njit(cache=True)
def _color_sort(
    cand_mask: np.uint64, compat: np.ndarray, weights: np.ndarray, order: np.ndarray, bounds: np.ndarray
) -> int:
    verts = np.empty(64, np.int64)
    count = 0
    tmp = cand_mask
    one = np.uint64(1)

    while tmp:
        bit = tmp & (~tmp + one)
        tmp ^= bit
        verts[count] = _bit_index(bit)
        count += 1

    for i in range(count - 1):
        best = i
        best_deg = -1
        best_w = -1
        for j in range(i, count):
            v = verts[j]
            deg = _popcount(compat[v] & cand_mask)
            w = int(weights[v])
            if deg > best_deg or (deg == best_deg and w > best_w):
                best = j
                best_deg = deg
                best_w = w
        if best != i:
            t = verts[i]
            verts[i] = verts[best]
            verts[best] = t

    rem = cand_mask
    out = 0
    cumulative = 0

    while rem:
        avail = rem
        start = out
        color_max = 0

        for p in range(count):
            v = verts[p]
            bit = one << np.uint64(v)
            if avail & bit:
                order[out] = v
                out += 1
                rem &= ~bit
                avail &= ~(bit | compat[v])
                w = int(weights[v])
                if w > color_max:
                    color_max = w

        cumulative += color_max
        for p in range(start, out):
            bounds[p] = cumulative

    return out

@njit(cache=True)
def _solve_rec(cand_mask: np.uint64, compat: np.ndarray, weights: np.ndarray) -> tuple[int, np.uint64]:
    if cand_mask == 0:
        return 0, np.uint64(0)

    best_weight, best_mask = _greedy_lb(cand_mask, compat, weights)

    order = np.empty(64, np.int64)
    bounds = np.empty(64, np.int64)
    count = _color_sort(cand_mask, compat, weights, order, bounds)

    local_cand = cand_mask
    one = np.uint64(1)

    for pos in range(count - 1, -1, -1):
        v = int(order[pos])
        bit = one << np.uint64(v)
        if (local_cand & bit) == 0:
            continue

        if bounds[pos] <= best_weight:
            return best_weight, best_mask

        new_cand = local_cand & compat[v]
        if new_cand == 0:
            new_weight = int(weights[v])
            new_mask = bit
        else:
            sub_weight, sub_mask = _solve_rec(new_cand, compat, weights)
            new_weight = int(weights[v]) + sub_weight
            new_mask = bit | sub_mask

        if new_weight > best_weight:
            best_weight = new_weight
            best_mask = new_mask

        local_cand &= ~bit

    return best_weight, best_mask

@njit(cache=True)
def _solve_local_exact(compat: np.ndarray, weights: np.ndarray) -> np.uint64:
    k = len(weights)
    if k == 0:
        return np.uint64(0)
    all_mask = (np.uint64(1) << np.uint64(k)) - np.uint64(1)
    _, mask = _solve_rec(all_mask, compat, weights)
    return mask

class Solver:
    def __init__(self) -> None:
        compat = np.zeros(1, dtype=np.uint64)
        weights = np.ones(1, dtype=np.int64)
        _solve_local_exact(compat, weights)

    def solve(self, problem, **kwargs) -> Any:
        adj_matrix = problem["adj_matrix"]
        weights_in = problem["weights"]
        n = len(weights_in)
        if n == 0:
            return []

        orig_idx = [i for i, w in enumerate(weights_in) if w > 0]
        if not orig_idx:
            return []

        m = len(orig_idx)
        weights = tuple(weights_in[i] for i in orig_idx)

        adj = [0] * m
        for i_local, i_orig in enumerate(orig_idx):
            row = adj_matrix[i_orig]
            bits = 0
            for j_local, j_orig in enumerate(orig_idx):
                if row[j_orig]:
                    bits |= 1 << j_local
            adj[i_local] = bits

        def bit_idx(bit: int) -> int:
            return bit.bit_length() - 1

        def vertices(mask: int) -> list[int]:
            out: list[int] = []
            cur = mask
            while cur:
                bit = cur & -cur
                cur ^= bit
                out.append(bit.bit_length() - 1)
            return out

        def connected_components(mask: int) -> list[int]:
            comps: list[int] = []
            remaining = mask
            while remaining:
                seed = remaining & -remaining
                remaining ^= seed
                comp = seed
                frontier = seed
                while frontier:
                    bit = frontier & -frontier
                    frontier ^= bit
                    v = bit.bit_length() - 1
                    nbrs = adj[v] & remaining
                    if nbrs:
                        remaining ^= nbrs
                        frontier |= nbrs
                        comp |= nbrs
                comps.append(comp)
            return comps

        def comp_info(comp_mask: int, verts: list[int]) -> tuple[int, int]:
            edges2 = 0
            max_deg = 0
            for v in verts:
                deg = (adj[v] & comp_mask).bit_count()
                edges2 += deg
                if deg > max_deg:
                    max_deg = deg
            return edges2 >> 1, max_deg

        def solve_tree(comp_mask: int, verts: list[int]) -> int:
            root = verts[0]
            children = {v: [] for v in verts}
            order = [root]
            stack = [root]
            seen = 1 << root

            while stack:
                v = stack.pop()
                nbrs = adj[v] & comp_mask
                while nbrs:
                    bit = nbrs & -nbrs
                    nbrs ^= bit
                    u = bit_idx(bit)
                    if seen & bit:
                        continue
                    seen |= bit
                    children[v].append(u)
                    order.append(u)
                    stack.append(u)

            inc: dict[int, int] = {}
            exc: dict[int, int] = {}
            for v in reversed(order):
                take = weights[v]
                skip = 0
                for u in children[v]:
                    take += exc[u]
                    skip += inc[u] if inc[u] >= exc[u] else exc[u]
                inc[v] = take
                exc[v] = skip

            sel = 0
            stack2: list[tuple[int, bool]] = [(root, False)]
            while stack2:
                v, parent_taken = stack2.pop()
                take_v = False
                if not parent_taken and inc[v] >= exc[v]:
                    take_v = True
                    sel |= 1 << v
                for u in children[v]:
                    stack2.append((u, take_v))
            return sel

        def solve_small_exact(comp_mask: int, verts: list[int]) -> int:
            k = len(verts)
            pos = {v: i for i, v in enumerate(verts)}
            local_weights = np.empty(k, dtype=np.int64)
            compat = np.empty(k, dtype=np.uint64)

            for i, v in enumerate(verts):
                local_weights[i] = weights[v]

            for i, v in enumerate(verts):
                mask = comp_mask & ~(adj[v] | (1 << v))
                local = 0
                tmp = mask
                while tmp:
                    bit = tmp & -tmp
                    tmp ^= bit
                    local |= 1 << pos[bit_idx(bit)]
                compat[i] = np.uint64(local)

            local_sel = int(_solve_local_exact(compat, local_weights))
            sel = 0
            for i, v in enumerate(verts):
                if (local_sel >> i) & 1:
                    sel |= 1 << v
            return sel

        all_mask = (1 << m) - 1
        chosen_mask = 0
        residual_verts: list[int] = []

        for comp in connected_components(all_mask):
            verts = vertices(comp)
            k = len(verts)
            if k == 1:
                chosen_mask |= comp
                continue

            e, max_deg = comp_info(comp, verts)

            if e == 0:
                chosen_mask |= comp
                continue

            if e == k * (k - 1) // 2:
                best_v = max(verts, key=lambda v: weights[v])
                chosen_mask |= 1 << best_v
                continue

            if e == k - 1:
                chosen_mask |= solve_tree(comp, verts)
                continue

            if k <= 63:
                chosen_mask |= solve_small_exact(comp, verts)
                continue

            if max_deg <= 2 and e == k:
                # Weighted cycle DP via two path cases.
                deg2 = [v for v in verts if (adj[v] & comp).bit_count() == 2]
                start = deg2[0] if deg2 else verts[0]
                nbr_bits = adj[start] & comp
                b1 = nbr_bits & -nbr_bits
                b2 = nbr_bits ^ b1
                n1 = bit_idx(b1)
                n2 = bit_idx(b2)

                order1 = [start]
                prev = -1
                cur = n1
                while cur != start:
                    order1.append(cur)
                    nbrs = adj[cur] & comp
                    nxt = -1
                    tmp = nbrs
                    while tmp:
                        bit = tmp & -tmp
                        tmp ^= bit
                        u = bit_idx(bit)
                        if u != prev:
                            nxt = u
                            break
                    prev, cur = cur, nxt

                def path_dp(order_list: list[int]) -> tuple[int, int]:
                    r = len(order_list)
                    if r == 0:
                        return 0, 0
                    if r == 1:
                        return weights[order_list[0]], 1 << order_list[0]
                    dp0_w, dp0_m = 0, 0
                    dp1_w, dp1_m = weights[order_list[0]], 1 << order_list[0]
                    for t in range(1, r):
                        v = order_list[t]
                        take_w = dp0_w + weights[v]
                        take_m = dp0_m | (1 << v)
                        if dp1_w >= dp0_w:
                            skip_w, skip_m = dp1_w, dp1_m
                        else:
                            skip_w, skip_m = dp0_w, dp0_m
                        dp0_w, dp0_m = skip_w, skip_m
                        dp1_w, dp1_m = take_w, take_m
                    if dp1_w >= dp0_w:
                        return dp1_w, dp1_m
                    return dp0_w, dp0_m

                w0, m0 = path_dp(order1[1:])
                w1, m1 = path_dp(order1[2:-1])
                w1 += weights[start]
                m1 |= 1 << start
                chosen_mask |= m1 if w1 >= w0 else m0
                continue

            residual_verts.extend(verts)

        if residual_verts:
            residual_mask = 0
            for v in residual_verts:
                residual_mask |= 1 << v

            model = cp_model.CpModel()
            x = [model.NewBoolVar(f"x_{i}") for i in range(len(residual_verts))]

            loc = [-1] * m
            for i, v in enumerate(residual_verts):
                loc[v] = i

            for i, v in enumerate(residual_verts):
                nbrs = adj[v] & residual_mask
                while nbrs:
                    bit = nbrs & -nbrs
                    nbrs ^= bit
                    u = bit_idx(bit)
                    j = loc[u]
                    if j > i:
                        model.Add(x[i] + x[j] <= 1)

            model.Maximize(sum(weights[v] * x[i] for i, v in enumerate(residual_verts)))

            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.linearization_level = 0
            solver.parameters.cp_model_probing_level = 0
            solver.parameters.symmetry_level = 0

            status = solver.Solve(model)
            if status == cp_model.OPTIMAL:
                for i, v in enumerate(residual_verts):
                    if solver.Value(x[i]):
                        chosen_mask |= 1 << v

        result: list[int] = []
        cur = chosen_mask
        while cur:
            bit = cur & -cur
            cur ^= bit
            result.append(orig_idx[bit_idx(bit)])
        result.sort()
        return result