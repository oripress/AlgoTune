from __future__ import annotations

from typing import Any, List, Optional, Tuple

class Solver:
    """
    Exact graph coloring (minimum #colors) using:
      - fast bitset adjacency representation (Python ints)
      - dominance reduction (neighborhood subset)
      - maximum clique lower bound (greedy + optional exact Tomita-style B&B)
      - DSATUR/MRV backtracking to decide k-colorability, seeded with a clique

    Returns a list of colors in 1..k for all original vertices.
    """

    def solve(self, problem: List[List[int]], **kwargs: Any) -> List[int]:
        n = len(problem)
        if n == 0:
            return []

        # --- Build adjacency bitmasks on original vertex set ---
        adj0 = [0] * n
        for i, row in enumerate(problem):
            m = 0
            for j, v in enumerate(row):
                if v:
                    m |= 1 << j
            adj0[i] = m & ~(1 << i)

        # Quick trivial cases
        any_edges = 0
        for a in adj0:
            any_edges |= a
        if any_edges == 0:
            return [1] * n

        deg0 = [a.bit_count() for a in adj0]

        # --- Dominance reduction (safe for coloring) ---
        # If N(u) ⊆ N(v) and (deg(v)>deg(u) or v>u), then u can take v's color.
        # Skip on very large n where it's unlikely to help and costs O(n^2).
        dom = list(range(n))
        if n <= 260:
            asc = sorted(range(n), key=lambda x: (deg0[x], x))
            desc = sorted(range(n), key=lambda x: (-deg0[x], -x))
            for u in asc:
                mu = adj0[u]
                du = deg0[u]
                for v in desc:
                    if v == u:
                        continue
                    dv = deg0[v]
                    if dv < du:
                        break
                    if dv == du and v < u:
                        continue  # enforce acyclic dominance
                    if (mu & ~adj0[v]) == 0:
                        dom[u] = v
                        break

            # Path compress
            for i in range(n):
                x = i
                while dom[x] != x:
                    dom[x] = dom[dom[x]]
                    x = dom[x]
                dom[i] = x

        roots = sorted(set(dom))
        nr = len(roots)
        if nr == 1:
            return [1] * n

        pos0 = [-1] * n
        for idx, r in enumerate(roots):
            pos0[r] = idx

        roots_mask0 = 0
        for r in roots:
            roots_mask0 |= 1 << r

        # Reduced graph adjacency
        adj = [0] * nr
        for idx, r in enumerate(roots):
            neigh = adj0[r] & roots_mask0
            mm = 0
            while neigh:
                b = neigh & -neigh
                j = b.bit_length() - 1
                mm |= 1 << pos0[j]
                neigh ^= b
            adj[idx] = mm

        deg = [a.bit_count() for a in adj]
        all_vertices_mask = (1 << nr) - 1

        # Complete graph detection
        complete = True
        for d in deg:
            if d != nr - 1:
                complete = False
                break
        if complete:
            root_color = list(range(1, nr + 1))
            out = [0] * n
            for i in range(n):
                out[i] = root_color[pos0[dom[i]]]
            return out

        # --- Fast bipartite check (often settles to 2 colors) ---
        if nr <= 450:
            part = [0] * nr
            ok_bip = True
            for s in range(nr):
                if part[s]:
                    continue
                part[s] = 1
                stack = [s]
                while stack and ok_bip:
                    u = stack.pop()
                    nb = adj[u]
                    cu = part[u]
                    other = 3 - cu
                    while nb:
                        b = nb & -nb
                        v = b.bit_length() - 1
                        nb ^= b
                        pv = part[v]
                        if pv == 0:
                            part[v] = other
                            stack.append(v)
                        elif pv == cu:
                            ok_bip = False
                            break
                if not ok_bip:
                    break
            if ok_bip:
                root_color = [1 if part[i] == 1 else 2 for i in range(nr)]
                out = [0] * n
                for i in range(n):
                    out[i] = root_color[pos0[dom[i]]]
                return out

        # Upper bound for ending the k-loop: always Δ+1 colorable.
        max_deg = 0
        for d in deg:
            if d > max_deg:
                max_deg = d
        ub_max = min(nr, max_deg + 1)

        # --- Clique lower bound (greedy + optional exact) ---
        order_deg = sorted(range(nr), key=lambda v: deg[v], reverse=True)

        def greedy_clique_bits() -> int:
            cand = all_vertices_mask
            clique = 0
            for v in order_deg:
                if (cand >> v) & 1:
                    clique |= 1 << v
                    cand &= adj[v]
                    if cand == 0:
                        break
            return clique

        best_clique_bits = greedy_clique_bits()
        lb = best_clique_bits.bit_count()

        # Exact max clique using Tomita-style coloring bound (more selective to reduce overhead)
        if nr <= 90 and 0 < lb < ub_max:
            best_bits = best_clique_bits

            def color_sort(P: int) -> Tuple[List[int], List[int]]:
                order: List[int] = []
                colors: List[int] = []
                color = 0
                while P:
                    color += 1
                    Q = P
                    while Q:
                        vb = Q & -Q
                        v = vb.bit_length() - 1
                        Q ^= vb
                        P ^= vb
                        order.append(v)
                        colors.append(color)
                        Q &= ~adj[v]  # keep independent set for this color
                return order, colors

            best_size = lb

            def expand(Csize: int, P: int) -> None:
                nonlocal best_bits, best_size
                if not P:
                    if Csize > best_size:
                        best_size = Csize
                        best_bits = current_clique[0]
                    return
                order, cols = color_sort(P)
                for i in range(len(order) - 1, -1, -1):
                    if Csize + cols[i] <= best_size:
                        return
                    v = order[i]
                    vb = 1 << v
                    current_clique[0] |= vb
                    expand(Csize + 1, P & adj[v])
                    current_clique[0] ^= vb
                    P ^= vb

            current_clique = [0]
            expand(0, all_vertices_mask)
            best_clique_bits = best_bits
            lb = best_size

        clique_vertices: List[int] = []
        tmpc = best_clique_bits
        while tmpc:
            b = tmpc & -tmpc
            clique_vertices.append(b.bit_length() - 1)
            tmpc ^= b
        # Heuristic: seed clique in descending degree order (slightly tighter propagation)
        clique_vertices.sort(key=lambda v: deg[v], reverse=True)

        # --- DSATUR/MRV k-colorability backtracking (exact) ---
        def k_colorable(k: int) -> Optional[List[int]]:
            fullmask = (1 << k) - 1
            colors = [0] * nr
            neigh = [0] * nr  # bitmask of colors present among colored neighbors
            uncolored = all_vertices_mask

            # Seed with clique: force distinct colors 1..|Q|
            if clique_vertices:
                if len(clique_vertices) > k:
                    return None
                for ci, v in enumerate(clique_vertices, start=1):
                    cbit = 1 << (ci - 1)
                    colors[v] = ci
                    uncolored ^= 1 << v
                    nb = adj[v] & uncolored
                    while nb:
                        b = nb & -nb
                        w = b.bit_length() - 1
                        nb ^= b
                        neigh[w] |= cbit

            adj_local = adj
            deg_local = deg

            def dfs(uncol: int) -> bool:
                if uncol == 0:
                    return True

                best_v = -1
                best_avail = 0
                best_mrv = 1 << 30
                best_sat = -1
                best_deg2 = -1

                tmp = uncol
                while tmp:
                    vb = tmp & -tmp
                    v = vb.bit_length() - 1
                    tmp ^= vb

                    used = neigh[v]
                    avail = fullmask & ~used
                    ac = avail.bit_count()
                    if ac == 0:
                        return False
                    if ac < best_mrv:
                        best_mrv = ac
                        best_v = v
                        best_avail = avail
                        best_sat = used.bit_count()
                        best_deg2 = deg_local[v]
                        if ac == 1:
                            break
                    elif ac == best_mrv:
                        sat = used.bit_count()
                        dv = deg_local[v]
                        if sat > best_sat or (sat == best_sat and dv > best_deg2):
                            best_v = v
                            best_avail = avail
                            best_sat = sat
                            best_deg2 = dv

                v = best_v
                vbit = 1 << v
                uncol2 = uncol ^ vbit

                avail = best_avail
                while avail:
                    cbit = avail & -avail
                    avail ^= cbit
                    colors[v] = cbit.bit_length()

                    changed: List[Tuple[int, int]] = []
                    nb = adj_local[v] & uncol2
                    while nb:
                        b = nb & -nb
                        w = b.bit_length() - 1
                        nb ^= b
                        old = neigh[w]
                        new = old | cbit
                        if new != old:
                            neigh[w] = new
                            changed.append((w, old))

                    if dfs(uncol2):
                        return True

                    for w, old in changed:
                        neigh[w] = old
                    colors[v] = 0

                return False

            if dfs(uncolored):
                return colors
            return None

        # Try increasing k from LB to a guaranteed UB.
        best_solution = None
        for k in range(lb, ub_max + 1):
            sol = k_colorable(k)
            if sol is not None:
                best_solution = sol
                break

        if best_solution is None:
            # Should not happen: k=Δ+1 always feasible.
            # Keep validator-safe fallback: color each vertex uniquely.
            best_solution = list(range(1, nr + 1))

        # Map back through dominators to original nodes
        out = [0] * n
        for i in range(n):
            out[i] = best_solution[pos0[dom[i]]]

        return out