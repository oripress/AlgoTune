from __future__ import annotations

import sys
from typing import Dict, List

class Solver:
    def __init__(self) -> None:
        sys.setrecursionlimit(10000)

    def solve(self, problem: list[list[int]], **kwargs) -> list[int]:
        n = len(problem)
        if n == 0:
            return []

        adj = [0] * n
        for i, row in enumerate(problem):
            mask = 0
            for j, val in enumerate(row):
                if val and i != j:
                    mask |= 1 << j
            adj[i] = mask

        all_mask = (1 << n) - 1

        def iter_bits(mask: int):
            while mask:
                lb = mask & -mask
                yield lb.bit_length() - 1
                mask ^= lb

        def bits_list(mask: int) -> List[int]:
            return list(iter_bits(mask))

        def components(mask: int) -> List[int]:
            rem = mask
            comps: List[int] = []
            while rem:
                seed = rem & -rem
                rem ^= seed
                comp = 0
                stack = seed
                while stack:
                    lb = stack & -stack
                    stack ^= lb
                    v = lb.bit_length() - 1
                    comp |= lb
                    nbrs = adj[v] & rem
                    if nbrs:
                        stack |= nbrs
                        rem ^= nbrs
                comps.append(comp)
            return comps

        def bipartite_coloring(mask: int) -> Dict[int, int] | None:
            side = [-1] * n
            rem = mask
            while rem:
                lb = rem & -rem
                rem ^= lb
                s = lb.bit_length() - 1
                side[s] = 0
                stack = [s]
                while stack:
                    u = stack.pop()
                    nbrs = adj[u] & mask
                    while nbrs:
                        x = nbrs & -nbrs
                        nbrs ^= x
                        v = x.bit_length() - 1
                        if side[v] == -1:
                            side[v] = side[u] ^ 1
                            rem &= ~x
                            stack.append(v)
                        elif side[v] == side[u]:
                            return None
            return {v: side[v] + 1 for v in iter_bits(mask)}

        def greedy_clique(mask: int, deg: List[int]) -> List[int]:
            order = sorted(iter_bits(mask), key=lambda v: deg[v], reverse=True)
            best: List[int] = []
            best_len = 0
            for start in order:
                if deg[start] + 1 <= best_len:
                    break
                clique = [start]
                cand = adj[start] & mask
                while cand:
                    best_v = -1
                    best_score = -1
                    scan = cand
                    while scan:
                        lb = scan & -scan
                        scan ^= lb
                        v = lb.bit_length() - 1
                        score = (adj[v] & cand).bit_count()
                        if score > best_score:
                            best_score = score
                            best_v = v
                    clique.append(best_v)
                    cand &= adj[best_v]
                if len(clique) > best_len:
                    best = clique
                    best_len = len(clique)
            return best

        def greedy_dsatur(mask: int, deg: List[int]) -> tuple[int, Dict[int, int]]:
            sat_mask = [0] * n
            sat_count = [0] * n
            color: Dict[int, int] = {}
            used = 0
            uncolored = mask

            while uncolored:
                best_v = -1
                best_sat = -1
                best_deg = -1
                scan = uncolored
                while scan:
                    lb = scan & -scan
                    scan ^= lb
                    v = lb.bit_length() - 1
                    sv = sat_count[v]
                    dv = deg[v]
                    if sv > best_sat or (sv == best_sat and dv > best_deg):
                        best_v = v
                        best_sat = sv
                        best_deg = dv

                forbidden = sat_mask[best_v]
                c = 0
                while forbidden & (1 << c):
                    c += 1
                color[best_v] = c + 1
                if c + 1 > used:
                    used = c + 1

                bit = 1 << c
                uncolored ^= 1 << best_v
                nbrs = adj[best_v] & uncolored
                while nbrs:
                    lb = nbrs & -nbrs
                    nbrs ^= lb
                    w = lb.bit_length() - 1
                    if not (sat_mask[w] & bit):
                        sat_mask[w] |= bit
                        sat_count[w] += 1

            return used, color

        def reduce_dominated(mask: int) -> tuple[int, List[int]]:
            parent = list(range(n))
            active = mask
            changed = True
            while changed:
                changed = False
                verts = bits_list(active)
                removed = 0
                m = len(verts)
                for i in range(m):
                    u = verts[i]
                    if removed & (1 << u):
                        continue
                    nu = adj[u] & active
                    for j in range(i + 1, m):
                        v = verts[j]
                        if removed & (1 << v):
                            continue
                        nv = adj[v] & active
                        if nu & ~nv == 0:
                            parent[u] = v
                            removed |= 1 << u
                            changed = True
                            break
                        if nv & ~nu == 0:
                            parent[v] = u
                            removed |= 1 << v
                            changed = True
                if removed:
                    active &= ~removed
            return active, parent

        def solve_connected(mask: int) -> Dict[int, int]:
            verts = bits_list(mask)
            m = len(verts)
            if m <= 1:
                return {v: 1 for v in verts}

            deg = [0] * n
            degree_sum = 0
            for v in verts:
                dv = (adj[v] & mask).bit_count()
                deg[v] = dv
                degree_sum += dv
            e = degree_sum >> 1

            if e == 0:
                return {v: 1 for v in verts}
            if e == m * (m - 1) // 2:
                return {v: i + 1 for i, v in enumerate(verts)}

            if e <= (m * m) // 4:
                bip = bipartite_coloring(mask)
                if bip is not None:
                    return bip

            if 4 <= m <= 80:
                reduced, parent = reduce_dominated(mask)
                if reduced != mask:
                    base = solve_subgraph(reduced)

                    def find_root(x: int) -> int:
                        path = []
                        while parent[x] != x:
                            path.append(x)
                            x = parent[x]
                        for y in path:
                            parent[y] = x
                        return x

                    return {v: base[find_root(v)] for v in verts}

            ub, greedy_color = greedy_dsatur(mask, deg)
            clique = greedy_clique(mask, deg)
            lb = len(clique)
            if lb == ub:
                return greedy_color

            color_of = [0] * n
            sat_mask = [0] * n
            sat_count = [0] * n
            best_k = ub
            best_assign = [greedy_color[v] for v in verts]
            target_k = lb
            stop = False

            uncolored = mask
            used = 0
            for c, u in enumerate(clique):
                color_of[u] = c + 1
                used = c + 1
                uncolored ^= 1 << u

            for c, u in enumerate(clique):
                bit = 1 << c
                nbrs = adj[u] & uncolored
                while nbrs:
                    lb2 = nbrs & -nbrs
                    nbrs ^= lb2
                    w = lb2.bit_length() - 1
                    if not (sat_mask[w] & bit):
                        sat_mask[w] |= bit
                        sat_count[w] += 1

            def search(u_mask: int, used_colors: int) -> None:
                nonlocal best_k, best_assign, stop
                if stop or used_colors >= best_k:
                    return
                if u_mask == 0:
                    best_k = used_colors
                    best_assign = [color_of[v] for v in verts]
                    if best_k == target_k:
                        stop = True
                    return

                best_v = -1
                best_sat = -1
                best_deg2 = -1
                scan = u_mask
                while scan:
                    lb2 = scan & -scan
                    scan ^= lb2
                    v = lb2.bit_length() - 1
                    sv = sat_count[v]
                    dv = (adj[v] & u_mask).bit_count()
                    if sv > best_sat or (sv == best_sat and dv > best_deg2):
                        best_v = v
                        best_sat = sv
                        best_deg2 = dv

                if best_sat == used_colors and used_colors + 1 >= best_k:
                    return

                forbidden = sat_mask[best_v]
                next_mask = u_mask ^ (1 << best_v)

                avail = ((1 << used_colors) - 1) & ~forbidden
                while avail:
                    color_bit = avail & -avail
                    avail ^= color_bit
                    color_of[best_v] = color_bit.bit_length()
                    changed: List[int] = []
                    nbrs = adj[best_v] & next_mask
                    while nbrs:
                        lb2 = nbrs & -nbrs
                        nbrs ^= lb2
                        w = lb2.bit_length() - 1
                        if not (sat_mask[w] & color_bit):
                            sat_mask[w] |= color_bit
                            sat_count[w] += 1
                            changed.append(w)
                    search(next_mask, used_colors)
                    for w in changed:
                        sat_mask[w] ^= color_bit
                        sat_count[w] -= 1
                    color_of[best_v] = 0
                    if stop:
                        return

                if used_colors + 1 < best_k:
                    color_bit = 1 << used_colors
                    color_of[best_v] = used_colors + 1
                    changed = []
                    nbrs = adj[best_v] & next_mask
                    while nbrs:
                        lb2 = nbrs & -nbrs
                        nbrs ^= lb2
                        w = lb2.bit_length() - 1
                        if not (sat_mask[w] & color_bit):
                            sat_mask[w] |= color_bit
                            sat_count[w] += 1
                            changed.append(w)
                    search(next_mask, used_colors + 1)
                    for w in changed:
                        sat_mask[w] ^= color_bit
                        sat_count[w] -= 1
                    color_of[best_v] = 0

            search(uncolored, used)
            return {v: best_assign[i] for i, v in enumerate(verts)}

        def solve_subgraph(mask: int) -> Dict[int, int]:
            comps = components(mask)
            if len(comps) == 1:
                return solve_connected(mask)
            out: Dict[int, int] = {}
            for comp in comps:
                out.update(solve_connected(comp))
            return out

        result = solve_subgraph(all_mask)
        colors = [result[i] for i in range(n)]
        used = sorted(set(colors))
        remap = {c: i + 1 for i, c in enumerate(used)}
        return [remap[c] for c in colors]