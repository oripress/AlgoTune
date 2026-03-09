from __future__ import annotations

import sys
from typing import Any

def _greedy_clique(adj: list[int], starts: list[int]) -> int:
    best_mask = 0
    best_size = 0

    for start in starts:
        mask = 1 << start
        cand = adj[start]
        size = 1
        while cand:
            vb = cand & -cand
            cand &= cand - 1
            v = vb.bit_length() - 1
            mask |= vb
            size += 1
            cand &= adj[v]
        if size > best_size:
            best_size = size
            best_mask = mask

    return best_mask

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        n = len(problem)
        if n == 0:
            return []

        if n == 1:
            return [0]

        sys.setrecursionlimit(max(1000, n + 100))

        degrees = [sum(row) - (row[i] if i < len(row) else 0) for i, row in enumerate(problem)]
        perm = sorted(range(n), key=lambda i: (degrees[i], -i), reverse=True)
        pos = [0] * n
        for new_i, old_i in enumerate(perm):
            pos[old_i] = new_i

        adj = [0] * n
        new_deg = [0] * n

        for new_i, old_i in enumerate(perm):
            row = problem[old_i]
            bits = 0
            for old_j, val in enumerate(row):
                if val and old_j != old_i:
                    bits |= 1 << pos[old_j]
            adj[new_i] = bits
            new_deg[new_i] = bits.bit_count()

        if all(d == n - 1 for d in new_deg):
            return list(range(n))

        trial_count = min(n, 24)
        best_mask = _greedy_clique(adj, list(range(trial_count)))
        if best_mask == 0:
            best_mask = 1
        best_size = best_mask.bit_count()

        active = 0
        for v, d in enumerate(new_deg):
            if d >= best_size:
                active |= 1 << v

        if active == 0:
            return sorted(perm[v] for v in range(n) if (best_mask >> v) & 1)

        bit_length = int.bit_length

        def color_sort(p: int) -> tuple[list[int], list[int]]:
            order: list[int] = []
            bounds: list[int] = []
            color = 0
            while p:
                color += 1
                q = p
                while q:
                    vb = q & -q
                    v = bit_length(vb) - 1
                    order.append(v)
                    bounds.append(color)
                    p ^= vb
                    q ^= vb
                    q &= ~adj[v]
            return order, bounds

        def expand(p: int, size: int, clique_mask: int) -> None:
            nonlocal best_size, best_mask

            if not p:
                if size > best_size:
                    best_size = size
                    best_mask = clique_mask
                return

            if size + p.bit_count() <= best_size:
                return

            order, bounds = color_sort(p)

            for i in range(len(order) - 1, -1, -1):
                if size + bounds[i] <= best_size:
                    return

                v = order[i]
                vb = 1 << v
                new_size = size + 1
                new_mask = clique_mask | vb
                new_p = p & adj[v]

                if not new_p:
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = new_mask
                else:
                    expand(new_p, new_size, new_mask)

                p ^= vb
                if new_size + p.bit_count() <= best_size:
                    return

        expand(active, 0, 0)

        return sorted(perm[v] for v in range(n) if (best_mask >> v) & 1)