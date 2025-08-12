from __future__ import annotations

from typing import Any, List

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the Maximum Independent Set problem exactly.

        Approach:
        - Reduce MIS to Maximum Clique on the complement graph.
        - Use a Tomita-style branch-and-bound with greedy coloring (bitset-based).
        - Reorder vertices by ascending degree in the original graph to improve search.

        Input: problem is a 2D adjacency matrix (list of lists) with 0/1 entries, symmetric.
        Output: list of node indices forming a maximum independent set.
        """
        A = problem
        if A is None:
            return []
        n = len(A)
        if n == 0:
            return []

        # Normalize input rows and determine degrees in the original graph (ignore diagonal)
        deg = [0] * n
        for i in range(n):
            row = A[i]
            if not isinstance(row, list):
                row = list(row)
                A[i] = row
            try:
                s = sum(row)
            except TypeError:
                s = 0
                for v in row:
                    s += 1 if v else 0
            if i < len(row) and row[i]:
                s -= 1
            deg[i] = int(s)

        # Permutation: sort by ascending degree in original graph (equivalently, descending in complement)
        perm = list(range(n))
        perm.sort(key=deg.__getitem__)
        pos = [0] * n
        for new_idx, old_idx in enumerate(perm):
            pos[old_idx] = new_idx
        rev = perm  # new_idx -> old_idx

        # Build complement graph adjacency as bitsets in the permuted ordering
        full_mask = (1 << n) - 1
        adj_c = [0] * n
        pow2_pos = [1 << pos[j] for j in range(n)]
        for new_i, old_i in enumerate(perm):
            row = A[old_i]
            nb = 0
            for old_j in range(n):
                if row[old_j]:
                    nb |= pow2_pos[old_j]
            # Complement: all except self and original neighbors
            adj_c[new_i] = (full_mask ^ (1 << new_i)) & ~nb

        # Edge cases quick returns
        if all(d == 0 for d in deg):
            return list(range(n))
        if all(d == n - 1 for d in deg):
            return [0]

        # Greedy initial clique in complement using order by descending complement degree
        comp_deg = [adj_c[i].bit_count() for i in range(n)]
        order_by_comp_deg = list(range(n))
        order_by_comp_deg.sort(key=lambda v: comp_deg[v], reverse=True)
        cand = full_mask
        greedy_mask = 0
        for v in order_by_comp_deg:
            v_bit = 1 << v
            if cand & v_bit:
                greedy_mask |= v_bit
                cand &= adj_c[v]
                if not cand:
                    break
        best_mask = greedy_mask
        best_size = greedy_mask.bit_count()

        # Tomita-style branch-and-bound with greedy coloring on bitsets
        def color_sort(P: int) -> tuple[List[int], List[int]]:
            adj = adj_c  # local binding
            order: List[int] = []
            bounds: List[int] = []
            order_append = order.append
            bounds_append = bounds.append
            p = P
            color = 0
            while p:
                color += 1
                q = p
                while q:
                    v_bit = q & -q
                    v = v_bit.bit_length() - 1
                    order_append(v)
                    bounds_append(color)
                    q &= ~(adj[v] | v_bit)
                    p &= ~v_bit
            return order, bounds

        # Recursive expansion
        def expand(P: int, cur_mask: int, cur_size: int) -> None:
            nonlocal best_size, best_mask
            if not P:
                if cur_size > best_size:
                    best_size = cur_size
                    best_mask = cur_mask
                return

            order, bounds = color_sort(P)
            # Global bound prune
            if cur_size + bounds[-1] <= best_size:
                return

            adj = adj_c  # local binding
            local_P = P
            # Traverse in reverse order (non-increasing bound) for stronger pruning
            idx = len(order) - 1
            while idx >= 0:
                v = order[idx]
                if cur_size + bounds[idx] <= best_size:
                    return
                v_bit = 1 << v
                local_P &= ~v_bit
                # Branch including v
                new_P = local_P & adj[v]
                new_size = cur_size + 1
                new_mask = cur_mask | v_bit
                if new_size > best_size:
                    best_size = new_size
                    best_mask = new_mask
                if new_P:
                    expand(new_P, new_mask, new_size)
                idx -= 1

        expand(full_mask, 0, 0)

        # Convert best clique in complement to MIS in original indexing
        result = []
        mask = best_mask
        result_append = result.append
        while mask:
            v_bit = mask & -mask
            new_idx = v_bit.bit_length() - 1
            result_append(rev[new_idx])
            mask ^= v_bit
        result.sort()
        return result