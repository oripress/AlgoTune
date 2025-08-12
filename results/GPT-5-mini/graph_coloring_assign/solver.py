from typing import List, Any
import sys

sys.setrecursionlimit(10000)

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> Any:
        """
        Graph coloring solver returning a list of colors (1..k) for each vertex.

        Strategy:
        - Build bitmask adjacency
        - Dominator (dominated-vertex) reduction
        - Tomita (Bron–Kerbosch with pivot) max-clique for a lower bound
        - Greedy largest-first for an upper bound
        - DSATUR-like branch-and-bound seeded with the clique
        """
        n = len(problem)
        if n == 0:
            return []

        # Build adjacency bitmasks and neighbor lists
        adj = [0] * n
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            row = problem[i]
            ln = min(len(row), n)
            mask = 0
            for j in range(ln):
                if i != j and row[j]:
                    mask |= 1 << j
                    neighbors[i].append(j)
            adj[i] = mask

        # Quick exit for empty graph
        if all(a == 0 for a in adj):
            return [1] * n

        # Dominator preprocessing removed to ensure correctness on all graphs.
        # Keeping a trivial dominator mapping (identity) avoids collapsing adjacent
        # vertices which could lead to invalid colorings (e.g., in complete graphs).
        dominator = list(range(n))
        remaining = set(range(n))
        rem_list = sorted(remaining)
        m = len(rem_list)

        # If everything reduced away, fallback to greedy on original graph
        if m == 0:
            order = sorted(range(n), key=lambda v: -len(neighbors[v]))
            greedy = [0] * n
            for v in order:
                used = set()
                for u in neighbors[v]:
                    if greedy[u]:
                        used.add(greedy[u])
                c = 1
                while c in used:
                    c += 1
                greedy[v] = c
            used_sorted = sorted(set(greedy))
            remap = {old: i + 1 for i, old in enumerate(used_sorted)}
            return [remap[c] for c in greedy]

        idx_map = {orig: i for i, orig in enumerate(rem_list)}

        # Build reduced adjacency (bitmask and neighbor lists)
        adj_red = [0] * m
        neighbors_red = [[] for _ in range(m)]
        for i, orig in enumerate(rem_list):
            mask = adj[orig]
            for j_orig in rem_list:
                if j_orig != orig and ((mask >> j_orig) & 1):
                    j = idx_map[j_orig]
                    neighbors_red[i].append(j)
                    adj_red[i] |= 1 << j
        deg_red = [len(neighbors_red[i]) for i in range(m)]

        # Greedy largest-first on reduced graph for an upper bound
        order = sorted(range(m), key=lambda v: -deg_red[v])
        greedy_red = [0] * m
        for v in order:
            used_mask = 0
            for u in neighbors_red[v]:
                cu = greedy_red[u]
                if cu:
                    used_mask |= 1 << (cu - 1)
            c = 1
            while used_mask & (1 << (c - 1)):
                c += 1
            greedy_red[v] = c
        ub = max(greedy_red) if m > 0 else 0
        best_coloring = greedy_red[:]
        best_k = ub

        # Tomita (Bron–Kerbosch with pivot) max clique on reduced graph for lower bound
        max_clique_mask = 0
        max_clique_size = 0
        if m > 0:
            all_mask = (1 << m) - 1
            adjm = adj_red

            def tomita():
                nonlocal max_clique_mask, max_clique_size

                def bk(R: int, P: int, X: int):
                    nonlocal max_clique_mask, max_clique_size
                    if P == 0 and X == 0:
                        rsize = R.bit_count()
                        if rsize > max_clique_size:
                            max_clique_size = rsize
                            max_clique_mask = R
                        return
                    # bound pruning
                    if R.bit_count() + P.bit_count() <= max_clique_size:
                        return
                    PX = P | X
                    # choose pivot with max neighbors in P
                    pivot = -1
                    max_nb = -1
                    tmp = PX
                    while tmp:
                        b = tmp & -tmp
                        u = b.bit_length() - 1
                        tmp ^= b
                        nb = (P & adjm[u]).bit_count()
                        if nb > max_nb:
                            max_nb = nb
                            pivot = u
                    if pivot == -1:
                        candidates = P
                    else:
                        candidates = P & ~adjm[pivot]
                    tmp = candidates
                    while tmp:
                        b = tmp & -tmp
                        v = b.bit_length() - 1
                        tmp ^= b
                        bk(R | (1 << v), P & adjm[v], X & adjm[v])
                        P &= ~(1 << v)
                        X |= 1 << v

                bk(0, all_mask, 0)

            tomita()
            lb = max_clique_size
        else:
            lb = 0

        # If clique lower bound equals greedy upper bound, return greedy mapping
        if lb == ub:
            c_red = greedy_red
            colors_full = [0] * n
            for v in range(n):
                root = v
                while dominator[root] != root:
                    root = dominator[root]
                if root in idx_map:
                    colors_full[v] = c_red[idx_map[root]]
                else:
                    colors_full[v] = 1
            used_sorted = sorted(set(colors_full))
            remap = {old: i + 1 for i, old in enumerate(used_sorted)}
            return [remap[c] for c in colors_full]

        # DSATUR-like branch-and-bound seeded with the clique
        colors_red = [0] * m
        clique_vertices = []
        tmp_mask = max_clique_mask
        while tmp_mask:
            b = tmp_mask & -tmp_mask
            v = b.bit_length() - 1
            clique_vertices.append(v)
            tmp_mask ^= b
        clique_vertices.sort()
        for i, v in enumerate(clique_vertices):
            colors_red[v] = i + 1
        used_colors = lb
        colored = len(clique_vertices)

        neighbor_color_mask = [0] * m
        for v in range(m):
            if colors_red[v]:
                bit = 1 << (colors_red[v] - 1)
                for u in neighbors_red[v]:
                    neighbor_color_mask[u] |= bit

        N = m
        NEIGH = neighbors_red
        DEG = deg_red

        best_k_local = best_k
        best_coloring_local = best_coloring[:]

        def choose_vertex():
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in range(N):
                if colors_red[v] == 0:
                    sat = neighbor_color_mask[v].bit_count()
                    if sat > best_sat or (sat == best_sat and DEG[v] > best_deg):
                        best_sat = sat
                        best_deg = DEG[v]
                        best_v = v
            return best_v

        def greedy_color_remaining(cur_used: int) -> int:
            temp_ncm = neighbor_color_mask[:]  # copy
            uncolored = [v for v in range(N) if colors_red[v] == 0]
            if not uncolored:
                return 0
            order_local = sorted(uncolored, key=lambda x: -DEG[x])
            used = cur_used
            add_used = 0
            for v in order_local:
                if used > 0:
                    allowed_mask = ((1 << used) - 1) & ~temp_ncm[v]
                else:
                    allowed_mask = 0
                if allowed_mask:
                    pos = (allowed_mask & -allowed_mask).bit_length() - 1
                    c = pos + 1
                else:
                    add_used += 1
                    used += 1
                    c = used
                bit = 1 << (c - 1)
                for u in NEIGH[v]:
                    temp_ncm[u] |= bit
            return add_used

        def dfs(colored_count: int, used_colors_count: int):
            nonlocal best_k_local, best_coloring_local
            if used_colors_count >= best_k_local:
                return
            if colored_count == N:
                best_k_local = used_colors_count
                best_coloring_local = colors_red[:]
                return
            add_needed = greedy_color_remaining(used_colors_count)
            if used_colors_count + add_needed >= best_k_local:
                return
            v = choose_vertex()
            if v == -1:
                return

            # try existing colors
            if used_colors_count > 0:
                allowed_existing = ((1 << used_colors_count) - 1) & ~neighbor_color_mask[v]
            else:
                allowed_existing = 0
            mask = allowed_existing
            while mask:
                b = mask & -mask
                mask ^= b
                pos = b.bit_length() - 1
                cnum = pos + 1
                colors_red[v] = cnum
                changed = []
                bit = 1 << pos
                for u in NEIGH[v]:
                    if colors_red[u] == 0 and not (neighbor_color_mask[u] & bit):
                        neighbor_color_mask[u] |= bit
                        changed.append(u)
                dfs(colored_count + 1, used_colors_count)
                for u in changed:
                    neighbor_color_mask[u] &= ~bit
                colors_red[v] = 0
                if best_k_local == lb:
                    return

            # try a new color (next unused)
            pos_new = used_colors_count
            cnum = used_colors_count + 1
            colors_red[v] = cnum
            changed = []
            bitnew = 1 << pos_new
            for u in NEIGH[v]:
                if colors_red[u] == 0 and not (neighbor_color_mask[u] & bitnew):
                    neighbor_color_mask[u] |= bitnew
                    changed.append(u)
            dfs(colored_count + 1, used_colors_count + 1)
            for u in changed:
                neighbor_color_mask[u] &= ~bitnew
            colors_red[v] = 0

        dfs(colored, used_colors)

        res_red = best_coloring_local if best_k_local < best_k else best_coloring

        # Map back to original graph through dominator chains
        colors_full = [0] * n
        for v in range(n):
            root = v
            # follow chain to a remaining root
            while dominator[root] != root:
                root = dominator[root]
            if root in idx_map:
                colors_full[v] = res_red[idx_map[root]]
            else:
                colors_full[v] = 1

        # Normalize so colors span 1..k
        used_vals = sorted(set(colors_full))
        remap = {old: new for new, old in enumerate(used_vals, start=1)}
        return [remap[c] for c in colors_full]