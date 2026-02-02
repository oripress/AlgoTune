from typing import Any, List

import sys

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> Any:
        """
        Exact Maximum Independent Set via Maximum Clique in the complement graph
        using a Tomita-style branch-and-bound with greedy coloring upper bounds.
        """
        n = len(problem)
        if n == 0:
            return []
        if n == 1:
            return [0]

        # Avoid recursion depth issues on some instances.
        if n + 50 > sys.getrecursionlimit():
            sys.setrecursionlimit(n + 200)

        bit_count = int.bit_count
        bit_length = int.bit_length

        all_mask = (1 << n) - 1

        # Build adjacency bitmasks for the original graph.
        adj_orig = [0] * n
        for i, row in enumerate(problem):
            m = 0
            for j, v in enumerate(row):
                if v:
                    m |= 1 << j
            adj_orig[i] = m

        # Forced vertices: isolated vertices in the original graph are always in an MIS.
        forced: List[int] = []
        forced_mask = 0
        for i, m in enumerate(adj_orig):
            if (m & (all_mask ^ (1 << i))) == 0:
                forced.append(i)
                forced_mask |= 1 << i

        if forced_mask == all_mask:
            return list(range(n))

        rem_mask = all_mask ^ forced_mask  # vertices to optimize over
        rem_n = bit_count(rem_mask)
        if rem_n <= 1:
            if rem_n == 1:
                v = bit_length(rem_mask & -rem_mask) - 1
                forced.append(v)
            return forced

        # Complement adjacency restricted to remaining vertices.
        comp_adj = [0] * n
        comp_deg = [0] * n
        for i in range(n):
            if (rem_mask >> i) & 1:
                cm = rem_mask & ~(adj_orig[i] | (1 << i))
                comp_adj[i] = cm
                comp_deg[i] = bit_count(cm)

        adj = comp_adj  # local alias for speed

        # Precompute "non-neighbors" in complement within rem_mask to avoid ~adj[v] (negative ints).
        non_nb = [0] * n
        for i in range(n):
            if (rem_mask >> i) & 1:
                non_nb[i] = rem_mask ^ adj[i]

        # ---------- Core max-clique on complement induced by mask0 ----------
        def max_clique_on_mask(mask0: int, use_pre_nb: bool) -> List[int]:
            adj_loc = adj
            comp_deg_loc = comp_deg

            # Quick greedy clique in complement to seed a good lower bound.
            # This version orders vertices once by (approx) degree and then builds a maximal clique.
            def greedy_clique(P: int) -> List[int]:
                verts: list[tuple[int, int]] = []
                Q = P
                if use_pre_nb:
                    while Q:
                        b = Q & -Q
                        Q ^= b
                        v = bit_length(b) - 1
                        verts.append((comp_deg_loc[v], v))
                else:
                    while Q:
                        b = Q & -Q
                        Q ^= b
                        v = bit_length(b) - 1
                        verts.append((bit_count(adj_loc[v] & P), v))

                verts.sort(reverse=True)

                clique: List[int] = []
                cand = P
                for _, v in verts:
                    if cand & (1 << v):
                        clique.append(v)
                        cand &= adj_loc[v]
                        if not cand:
                            break
                return clique

            best_clique = greedy_clique(mask0)
            best_size = len(best_clique)
            if use_pre_nb:
                nb = non_nb

                def color_sort(P: int) -> tuple[list[int], list[int]]:
                    k = bit_count(P)
                    order = [0] * k
                    colors = [0] * k
                    U = P
                    c = 0
                    idx = 0
                    while U:
                        c += 1
                        Q = U
                        while Q:
                            b = Q & -Q
                            Q ^= b
                            v = bit_length(b) - 1
                            U ^= b
                            Q &= nb[v]
                            order[idx] = v
                            colors[idx] = c
                            idx += 1
                    return order, colors

            else:

                def color_sort(P: int) -> tuple[list[int], list[int]]:
                    k = bit_count(P)
                    order = [0] * k
                    colors = [0] * k
                    U = P
                    c = 0
                    idx = 0
                    while U:
                        c += 1
                        Q = U
                        while Q:
                            b = Q & -Q
                            Q ^= b
                            v = bit_length(b) - 1
                            U ^= b
                            Q &= mask0 ^ (adj_loc[v] & mask0)
                            order[idx] = v
                            colors[idx] = c
                            idx += 1
                    return order, colors

            C = [0] * bit_count(mask0)

            def expand(P: int, csize: int) -> None:
                nonlocal best_size, best_clique

                if csize + bit_count(P) <= best_size:
                    return

                if not P:
                    if csize > best_size:
                        best_size = csize
                        best_clique = C[:csize]
                    return

                order, colors = color_sort(P)
                for idx in range(len(order) - 1, -1, -1):
                    if csize + colors[idx] <= best_size:
                        return
                    v = order[idx]
                    C[csize] = v
                    expand(P & adj_loc[v], csize + 1)
                    P ^= 1 << v

            expand(mask0, 0)
            return best_clique

        # ---------- Optional connected-component decomposition on original graph ----------
        # Only do work if rem_n is not tiny (avoid overhead on small graphs).
        if rem_n >= 60:
            # Check connectivity quickly.
            unseen = rem_mask
            b0 = unseen & -unseen
            unseen ^= b0
            stack = b0
            while stack:
                b = stack & -stack
                stack ^= b
                v = bit_length(b) - 1
                nbv = adj_orig[v] & unseen
                if nbv:
                    unseen ^= nbv
                    stack |= nbv
            if unseen:
                # Graph is disconnected: solve each connected component separately and union results.
                res = forced.copy()
                unseen = rem_mask
                while unseen:
                    b0 = unseen & -unseen
                    unseen ^= b0
                    comp = b0
                    stack = b0
                    while stack:
                        b = stack & -stack
                        stack ^= b
                        v = bit_length(b) - 1
                        nbv = adj_orig[v] & unseen
                        if nbv:
                            unseen ^= nbv
                            stack |= nbv
                            comp |= nbv
                    res.extend(max_clique_on_mask(comp, use_pre_nb=False))
                return res

        # Connected (or small): solve once on all remaining vertices using precomputed helpers.
        return forced + max_clique_on_mask(rem_mask, use_pre_nb=True)