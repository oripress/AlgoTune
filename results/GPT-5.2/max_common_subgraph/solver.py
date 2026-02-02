from __future__ import annotations

from typing import Any, List, Tuple

def _row_ones_mask(row: List[int]) -> int:
    """Bitmask of 1-entries in a 0/1 row."""
    mask = 0
    for j in range(len(row)):
        if row[j]:
            mask |= 1 << j
    return mask

def _max_clique_bitset(adj: List[int]) -> List[int]:
    """
    Exact maximum clique using branch-and-bound with greedy coloring upper bound.

    Graph is undirected with adjacency bitsets: adj[v] has neighbor bits over [0..V-1].
    Returns vertex indices of a maximum clique.
    """
    V = len(adj)
    if V == 0:
        return []

    fullmask = (1 << V) - 1
    # Precompute non-neighbors to avoid negative (~adj[v]) masks in coloring.
    nonadj = [fullmask ^ a for a in adj]  # includes self-bit; OK for coloring

    # Greedy heuristic clique for initial lower bound (dynamic degree).
    P = fullmask
    clique: List[int] = []
    while P:
        x = P
        best_v = -1
        best_d = -1
        while x:
            b = x & -x
            v = b.bit_length() - 1
            x ^= b
            d = (P & adj[v]).bit_count()
            if d > best_d:
                best_d = d
                best_v = v
        clique.append(best_v)
        P &= adj[best_v]

    best = clique
    best_size = len(best)

    adj_local = adj
    nonadj_local = nonadj

    def color_sort(Pset: int) -> tuple[list[int], list[int]]:
        # Greedy sequential coloring to get an upper bound.
        order: list[int] = []
        colors: list[int] = []
        U = Pset
        c = 0
        while U:
            c += 1
            Q = U
            while Q:
                b = Q & -Q
                v = b.bit_length() - 1
                Q ^= b
                order.append(v)
                colors.append(c)
                U ^= b
                Q &= U
                Q &= nonadj_local[v]
        return order, colors

    def expand(C: list[int], Pset: int) -> None:
        nonlocal best, best_size

        if not Pset:
            csz = len(C)
            if csz > best_size:
                best_size = csz
                best = C.copy()
            return

        # Very cheap bound before coloring.
        if len(C) + Pset.bit_count() <= best_size:
            return

        order, colors = color_sort(Pset)
        # Tomita: iterate in reverse coloring order.
        for idx in range(len(order) - 1, -1, -1):
            if len(C) + colors[idx] <= best_size:
                return
            v = order[idx]
            C.append(v)
            expand(C, Pset & adj_local[v])
            C.pop()
            Pset ^= 1 << v

    expand([], fullmask)
    return best

class Solver:
    def solve(
        self, problem: dict[str, list[list[int]]], **kwargs: Any
    ) -> List[Tuple[int, int]]:
        A = problem["A"]
        B = problem["B"]
        n = len(A)
        m = len(B)
        if n == 0 or m == 0:
            return []

        V = n * m
        allmask_m = (1 << m) - 1
        fullA = (1 << n) - 1

        # Row masks of ones in B.
        b1 = [_row_ones_mask(row) for row in B]

        # Precompute for each p:
        #  - all_blocks_p[j] = (q != p) bits shifted into G-block j
        #  - base0[p]: pattern for A-edge=0 everywhere (q != p AND B[p][q]==0), repeated over blocks
        #  - base1[p]: pattern for A-edge=1 everywhere (q != p AND B[p][q]==1), repeated over blocks
        blocks_all: list[list[int]] = []
        base0: list[int] = []
        base1: list[int] = []
        for p in range(m):
            all_q_notp = allmask_m ^ (1 << p)
            b1np = all_q_notp & b1[p]
            b0np = all_q_notp & ~b1np

            all_blocks_p: list[int] = []
            acc0 = 0
            acc1 = 0
            shift = 0
            for _ in range(n):
                ab = all_q_notp << shift
                all_blocks_p.append(ab)
                acc0 |= b0np << shift
                acc1 |= b1np << shift
                shift += m
            blocks_all.append(all_blocks_p)
            base0.append(acc0)
            base1.append(acc1)

        # Precompute A row masks and degrees (excluding self).
        a1mask: list[int] = [0] * n
        adeg: list[int] = [0] * n
        for i in range(n):
            mask = _row_ones_mask(A[i]) & ~(1 << i)
            a1mask[i] = mask
            adeg[i] = mask.bit_count()

        # Build association graph adjacency over vertices v=(i,p) -> index i*m+p
        adj: list[int] = [0] * V
        for i in range(n):
            shift_i = i * m
            block_mask_i = allmask_m << shift_i  # clear this block to enforce i != j
            ones_i = a1mask[i]
            zeros_i = ((fullA ^ (1 << i)) & ~ones_i)

            # Build by toggling fewer blocks (choose between ones or zeros).
            use_ones = adeg[i] <= (n - 1 - adeg[i])

            for p in range(m):
                ap = blocks_all[p]
                if use_ones:
                    neigh = base0[p] & ~block_mask_i
                    x = ones_i
                    while x:
                        b = x & -x
                        j = b.bit_length() - 1
                        x ^= b
                        neigh ^= ap[j]
                else:
                    neigh = base1[p] & ~block_mask_i
                    x = zeros_i
                    while x:
                        b = x & -x
                        j = b.bit_length() - 1
                        x ^= b
                        neigh ^= ap[j]
                adj[shift_i + p] = neigh

        clique = _max_clique_bitset(adj)
        return [(v // m, v % m) for v in clique]