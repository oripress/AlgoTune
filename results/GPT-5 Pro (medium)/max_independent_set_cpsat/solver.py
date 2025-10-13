from __future__ import annotations

from typing import Any, List

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve Maximum Independent Set using a fast exact Branch-and-Bound
        maximum clique algorithm on the complement graph with greedy coloring.

        Key optimizations:
        - Vertex renumbering by descending degree in the complement graph
          to improve greedy coloring and branching order.
        - Greedy initial clique on the complement graph for a strong lower bound.
        - Early popcount-based pruning before coloring.
        - Bitset-based operations for speed.

        Input:
            problem: 2D list (adjacency matrix with 0/1 entries, symmetric)

        Output:
            List of node indices in a maximum independent set.
        """
        A = problem
        n = len(A)
        if n == 0:
            return []

        # Build adjacency bitsets for the original graph
        mask_all = (1 << n) - 1
        adj_orig: List[int] = [0] * n
        for i in range(n):
            row = A[i]
            bits = 0
            for j, val in enumerate(row):
                if val:
                    bits |= (1 << j)
            bits &= ~(1 << i)  # ensure diagonal is zero
            adj_orig[i] = bits

        # Complement adjacency for clique search
        g_old: List[int] = [0] * n
        for i in range(n):
            vbit = 1 << i
            g_old[i] = (~adj_orig[i]) & (mask_all ^ vbit)

        # Static ordering by descending degree in complement graph.
        degc = [g_old[i].bit_count() for i in range(n)]
        perm = sorted(range(n), key=lambda i: (-degc[i], i))
        inv = [0] * n
        for new, old in enumerate(perm):
            inv[old] = new

        # Remap complement adjacency to the new ordering.
        g: List[int] = [0] * n
        for new_i, old_i in enumerate(perm):
            nb = g_old[old_i]
            mapped = 0
            while nb:
                lb = nb & -nb
                old_v = lb.bit_length() - 1
                nb &= ~lb
                mapped |= (1 << inv[old_v])
            g[new_i] = mapped & ~(1 << new_i)

        # Greedy initial clique on complement to get a strong lower bound.
        g_local = g
        cand = mask_all
        clique_bits = 0
        while cand:
            lb = cand & -cand
            v = lb.bit_length() - 1
            clique_bits |= lb
            cand &= g_local[v]
        best_set_bits = clique_bits
        best_size = best_set_bits.bit_count()

        # Greedy coloring to obtain upper bounds for pruning.
        def color_sort(P_bits: int):
            # Returns (order_list, color_bounds)
            order: List[int] = []
            bounds: List[int] = []
            U = P_bits
            g_loc = g_local
            append = order.append
            bappend = bounds.append
            color_num = 0
            while U:
                color_num += 1
                Q = U
                S = 0
                while Q:
                    lb = Q & -Q
                    v = lb.bit_length() - 1
                    append(v)
                    bappend(color_num)
                    Q &= ~lb
                    Q &= ~g_loc[v]
                    S |= lb
                U &= ~S
            return order, bounds

        # Branch and bound maximum clique with greedy coloring
        def expand(R_bits: int, P_bits: int, size_R: int):
            nonlocal best_size, best_set_bits
            # Simple size bound before coloring: remaining vertices cannot improve incumbent
            if size_R + P_bits.bit_count() <= best_size:
                return
            if P_bits == 0:
                if size_R > best_size:
                    best_size = size_R
                    best_set_bits = R_bits
                return

            order, bounds = color_sort(P_bits)
            P_local = P_bits
            for idx in range(len(order) - 1, -1, -1):
                if size_R + bounds[idx] <= best_size:
                    return
                v = order[idx]
                vbit = 1 << v
                expand(R_bits | vbit, P_local & g_local[v], size_R + 1)
                P_local &= ~vbit

        # Run the search on the renumbered graph
        expand(0, mask_all, 0)

        # Map the maximum clique in complement back to original vertex indices (MIS in original)
        res = []
        bs = best_set_bits
        while bs:
            lb = bs & -bs
            v_new = lb.bit_length() - 1
            res.append(perm[v_new])
            bs &= ~lb
        res.sort()
        return res