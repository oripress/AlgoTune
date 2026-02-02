from __future__ import annotations

from typing import Any

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

class Solver:
    __slots__ = ()

    def solve(self, problem: list[list[int]], **kwargs: Any) -> list[int]:
        n = len(problem)
        if n <= 0:
            return []
        if n == 1:
            return [0]

        bit_length = int.bit_length
        bit_count = int.bit_count

        # ---- Preprocessing: degree order + adjacency bitmasks ----
        # For larger graphs, using NumPy packbits is significantly faster than Python nested loops.
        use_numpy = _np is not None and n >= 80

        if use_numpy:
            A = _np.asarray(problem, dtype=_np.uint8)
            deg = A.sum(axis=1)
            order_arr = _np.argsort(-deg, kind="quicksort")
            order = order_arr.tolist()
            Ap = A[_np.ix_(order_arr, order_arr)]
            packed = _np.packbits(Ap, axis=1, bitorder="little")
            all_mask = (1 << n) - 1
            adj = [0] * n
            # Convert each packed row to a Python int mask.
            for i in range(n):
                m = int.from_bytes(packed[i].tobytes(), "little") & all_mask
                m &= ~(1 << i)
                adj[i] = m
        else:
            # Degree ordering (often reduces the search tree substantially).
            deg = [sum(row) for row in problem]
            order = sorted(range(n), key=deg.__getitem__, reverse=True)  # old index by new position
            pos = [0] * n  # new position of old index
            for new_i, old_i in enumerate(order):
                pos[old_i] = new_i

            # Build adjacency bitmasks in the reordered indexing.
            adj = [0] * n
            for new_i, old_i in enumerate(order):
                row = problem[old_i]
                m = 0
                for old_j, v in enumerate(row):
                    if v:
                        m |= 1 << pos[old_j]
                m &= ~(1 << new_i)
                adj[new_i] = m
            all_mask = (1 << n) - 1

        # Fast paths: complete graph / edgeless graph.
        # Complete: for all i, (adj[i] plus self) covers all vertices.
        for i, m in enumerate(adj):
            if (m | (1 << i)) != all_mask:
                break
        else:
            return order[:]  # already a permutation of all nodes
        if max(adj) == 0:
            return [order[0]]

        adj_local = adj  # local alias for speed

        # ---- Very cheap greedy lower bound (try a few starts among high-degree vertices) ----
        best_size = 1
        best_mask = 1  # always some vertex exists
        t = n if n < 16 else 16
        for s in range(t):
            cm = 1 << s
            P = adj_local[s]
            while P:
                b = P & -P
                v = bit_length(b) - 1
                cm |= b
                P &= adj_local[v]
            sz = bit_count(cm)
            if sz > best_size:
                best_size = sz
                best_mask = cm

        # ---- Greedy coloring for an upper bound (Tomita-style) ----
        def color_sort(P: int) -> tuple[list[int], list[int]]:
            order_v: list[int] = []
            colors: list[int] = []
            U = P
            color = 0
            append_v = order_v.append
            append_c = colors.append
            while U:
                color += 1
                Q = U
                while Q:
                    b = Q & -Q
                    Q ^= b
                    v = bit_length(b) - 1
                    U ^= b
                    Q &= ~adj_local[v]
                    append_v(v)
                    append_c(color)
            return order_v, colors

        # Branch-and-bound maximum clique search.
        def expand(P: int, c_size: int, c_mask: int) -> None:
            nonlocal best_size, best_mask

            if not P:
                if c_size > best_size:
                    best_size = c_size
                    best_mask = c_mask
                return

            # Quick bound by remaining vertices.
            if c_size + bit_count(P) <= best_size:
                return

            vs, cs = color_sort(P)

            # Explore in reverse order (highest colors first).
            for idx in range(len(vs) - 1, -1, -1):
                if c_size + cs[idx] <= best_size:
                    return
                v = vs[idx]
                bit = 1 << v
                expand(P & adj_local[v], c_size + 1, c_mask | bit)
                P &= ~bit
                if not P:
                    break
                if c_size + bit_count(P) <= best_size:
                    return

        expand(all_mask, 0, 0)

        # Convert best mask (new indices) back to original indices.
        res_new: list[int] = []
        m = best_mask
        while m:
            b = m & -m
            m ^= b
            res_new.append(bit_length(b) - 1)

        return [order[i] for i in res_new]