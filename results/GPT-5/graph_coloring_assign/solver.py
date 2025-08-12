from __future__ import annotations

from typing import Any, List, Optional, Tuple

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> List[int]:
        """
        Exact graph coloring using DSATUR-based branch-and-bound with bitset optimizations.

        Input:
            problem: adjacency matrix (symmetric, 0/1, zero diagonal)

        Output:
            List of colors (1..k) assigned to each vertex with minimum k.
        """
        A = problem
        n = len(A)
        if n == 0:
            return []

        # Build adjacency bitsets
        adj_mask = [0] * n
        for i in range(n):
            row = A[i]
            mask = 0
            for j in range(n):
                if row[j] and i != j:
                    mask |= (1 << j)
            adj_mask[i] = mask

        # Degree and helpers
        deg = [adj_mask[i].bit_count() for i in range(n)]
        order_deg_desc = sorted(range(n), key=lambda x: deg[x], reverse=True)
        one_bits = [1 << i for i in range(n)]

        # Greedy coloring (Welsh-Powell) for a fast upper bound
        def greedy_ub() -> Tuple[int, List[int]]:
            colors = [-1] * n
            color_sets: List[int] = []  # bitset of vertices per color
            for v in order_deg_desc:
                assigned = False
                av = adj_mask[v]
                bv = one_bits[v]
                for c, cset in enumerate(color_sets):
                    if (av & cset) == 0:
                        colors[v] = c
                        color_sets[c] |= bv
                        assigned = True
                        break
                if not assigned:
                    colors[v] = len(color_sets)
                    color_sets.append(bv)
            return len(color_sets), colors

        UB, greedy_coloring = greedy_ub()
        if UB <= 1:
            return [1] * n

        # Heuristic clique to obtain a lower bound and seed
        def heuristic_max_clique_nodes(limit_starts: int = 12) -> List[int]:
            order = sorted(range(n), key=lambda x: deg[x], reverse=True)
            best_clique: List[int] = []
            tried = 0
            for start in order:
                clique = [start]
                cand_mask = adj_mask[start]
                while cand_mask:
                    m = cand_mask
                    best_u = -1
                    best_d = -1
                    while m:
                        b = m & -m
                        u = b.bit_length() - 1
                        d = deg[u]
                        if d > best_d:
                            best_d = d
                            best_u = u
                        m ^= b
                    if best_u == -1:
                        break
                    clique.append(best_u)
                    cand_mask &= adj_mask[best_u]
                if len(clique) > len(best_clique):
                    best_clique = clique
                    if len(best_clique) >= UB:
                        break
                tried += 1
                if tried >= limit_starts:
                    break
            return best_clique if best_clique else [order[0]]

        clique_nodes = heuristic_max_clique_nodes()
        LB = max(1, len(clique_nodes))
        if LB >= UB:
            # Return normalized greedy coloring
            used = sorted(set(greedy_coloring))
            remap = {old: new for new, old in enumerate(used, start=1)}
            return [remap[c] for c in greedy_coloring]

        # DSATUR decision procedure for fixed k
        def colorable_with_k(k: int) -> Tuple[bool, List[int]]:
            maskK = (1 << k) - 1
            colors = [-1] * n  # 0-based colors
            forb = [0] * n  # bitset of forbidden colors (present among neighbors)
            sat_deg = [0] * n  # number of distinct neighbor colors
            # counts[c][u] = #neighbors of u colored with color c
            counts = [[0] * n for _ in range(k)]
            uncolored_mask = (1 << n) - 1
            used_colors_mask = 0
            num_used_colors = 0

            # Local aliases for speed
            adj = adj_mask
            oneb = one_bits
            deg_local = deg
            forb_local = forb
            sat_local = sat_deg
            cnt = counts

            # Pre-assign clique nodes to distinct colors to seed (touch only uncolored neighbors)
            r = min(len(clique_nodes), k)
            for c in range(r):
                v = clique_nodes[c]
                colors[v] = c
                used_colors_mask |= (1 << c)
                av = adj[v]
                cntc = cnt[c]
                # Only uncolored neighbors matter
                m = av & uncolored_mask
                while m:
                    b = m & -m
                    u = b.bit_length() - 1
                    prev = cntc[u]
                    cntc[u] = prev + 1
                    if prev == 0:
                        forb_local[u] |= (1 << c)
                        sat_local[u] += 1
                    m ^= b
                uncolored_mask &= ~oneb[v]
            num_used_colors = r

            # Select vertex with maximum saturation degree, tie-break by degree then MRV
            def select_vertex() -> int:
                m = uncolored_mask
                best_v = -1
                best_sd = -1
                best_d = -1
                best_av = 1 << 30
                while m:
                    b = m & -m
                    v = b.bit_length() - 1
                    sd = sat_local[v]
                    if sd > best_sd:
                        best_sd = sd
                        best_d = deg_local[v]
                        best_av = k - (forb_local[v] & maskK).bit_count()
                        best_v = v
                    else:
                        if sd == best_sd:
                            dv = deg_local[v]
                            if dv > best_d:
                                best_d = dv
                                best_av = k - (forb_local[v] & maskK).bit_count()
                                best_v = v
                            elif dv == best_d:
                                avc = k - (forb_local[v] & maskK).bit_count()
                                if avc < best_av:
                                    best_av = avc
                                    best_v = v
                    m ^= b
                return best_v

            def dfs() -> bool:
                nonlocal uncolored_mask, used_colors_mask, num_used_colors
                if uncolored_mask == 0:
                    return True

                v = select_vertex()
                av = adj[v]
                avail_mask = (~forb_local[v]) & maskK
                if avail_mask == 0:
                    return False

                # Try existing colors first
                existing = avail_mask & used_colors_mask
                x = existing
                while x:
                    bcol = x & -x
                    c = bcol.bit_length() - 1
                    colors[v] = c
                    cntc = cnt[c]
                    # Update only uncolored neighbors; record processed for undo; early fail if any neighbor loses last color
                    processed = 0
                    m = av & uncolored_mask
                    failed = False
                    while m:
                        b = m & -m
                        u = b.bit_length() - 1
                        prev = cntc[u]
                        cntc[u] = prev + 1
                        if prev == 0:
                            forb_u = forb_local[u] | (1 << c)
                            forb_local[u] = forb_u
                            sat_local[u] += 1
                            if (forb_u & maskK) == maskK:
                                failed = True
                        processed |= b
                        m ^= b
                        if failed:
                            break
                    old_uncolored = uncolored_mask
                    uncolored_mask &= ~oneb[v]

                    ok = False
                    if not failed:
                        ok = dfs()
                    if ok:
                        return True

                    # Undo
                    uncolored_mask = old_uncolored
                    m = processed
                    while m:
                        b = m & -m
                        u = b.bit_length() - 1
                        prev = cntc[u]
                        cntc[u] = prev - 1
                        if prev == 1:
                            forb_local[u] &= ~(1 << c)
                            sat_local[u] -= 1
                        m ^= b
                    colors[v] = -1
                    x ^= bcol

                # Introduce a single new color (symmetry breaking)
                if num_used_colors < k:
                    new_colors = avail_mask & (~used_colors_mask)
                    if new_colors:
                        bcol = new_colors & -new_colors
                        c = bcol.bit_length() - 1
                        colors[v] = c
                        prev_used_mask = used_colors_mask
                        prev_used_num = num_used_colors
                        used_colors_mask |= (1 << c)
                        num_used_colors = prev_used_num + 1

                        cntc = cnt[c]
                        # Update only uncolored neighbors; record processed for undo; early fail if any neighbor loses last color
                        processed = 0
                        m = av & uncolored_mask
                        failed = False
                        while m:
                            b = m & -m
                            u = b.bit_length() - 1
                            prev = cntc[u]
                            cntc[u] = prev + 1
                            if prev == 0:
                                forb_u = forb_local[u] | (1 << c)
                                forb_local[u] = forb_u
                                sat_local[u] += 1
                                if (forb_u & maskK) == maskK:
                                    failed = True
                            processed |= b
                            m ^= b
                            if failed:
                                break
                        old_uncolored = uncolored_mask
                        uncolored_mask &= ~oneb[v]

                        ok = False
                        if not failed:
                            ok = dfs()
                        if ok:
                            return True

                        # Undo
                        uncolored_mask = old_uncolored
                        m = processed
                        while m:
                            b = m & -m
                            u = b.bit_length() - 1
                            prev = cntc[u]
                            cntc[u] = prev - 1
                            if prev == 1:
                                forb_local[u] &= ~(1 << c)
                                sat_local[u] -= 1
                            m ^= b
                        colors[v] = -1
                        used_colors_mask = prev_used_mask
                        num_used_colors = prev_used_num

                return False

            feasible = dfs()
            if not feasible:
                return False, []

            # Normalize colors to 1..k' contiguous
            used = sorted(set(colors))
            remap = {old: new for new, old in enumerate(used, start=1)}
            sol = [remap[c] for c in colors]
            return True, sol

        best_sol: Optional[List[int]] = None
        for k in range(LB, UB + 1):
            feasible, sol = colorable_with_k(k)
            if feasible:
                best_sol = sol
                break

        if best_sol is None:
            # Fallback to normalized greedy coloring
            used = sorted(set(greedy_coloring))
            remap = {old: new for new, old in enumerate(used, start=1)}
            return [remap[c] for c in greedy_coloring]
        return best_sol