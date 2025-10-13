from __future__ import annotations

from typing import Any, List

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve Minimum Vertex Cover exactly.

        Strategy:
        - Compute a maximum independent set (MIS) using a fast exact maximum clique
          solver on the complement graph (Tomita-style with greedy coloring bound).
        - Return the complement of the MIS as the minimum vertex cover.

        Input:
            problem: adjacency matrix (symmetric 0/1 list of lists)

        Output:
            list of indices (0-based) forming a minimum vertex cover
        """
        # Parse adjacency matrix
        A = problem
        n = len(A)
        if n == 0:
            return []

        # Build adjacency bitsets for original graph
        # adj[i]: bitset of neighbors of i
        adj: List[int] = [0] * n
        for i in range(n):
            row = A[i]
            bits = 0
            # Ensure diagonal excluded, tolerate malformed diagonal
            for j in range(n):
                if i != j and row[j]:
                    bits |= 1 << j
            adj[i] = bits

        # Build complement graph adjacency bitsets for maximum clique computation
        full_mask = (1 << n) - 1
        comp_adj: List[int] = [0] * n
        for i in range(n):
            # Complement neighbors: all except i and original neighbors
            comp = (~adj[i]) & full_mask
            comp &= ~(1 << i)
            comp_adj[i] = comp

        # Greedy MIS (on original graph) to get an initial lower bound
        def greedy_mis_mask() -> int:
            cand = full_mask
            indep = 0
            while cand:
                v_bit = cand & -cand  # pick least significant bit
                v = (v_bit.bit_length() - 1)
                indep |= v_bit
                cand &= ~v_bit
                cand &= ~adj[v]
            return indep

        best_mis_mask = greedy_mis_mask()
        best_size = best_mis_mask.bit_count()

        # Max clique on complement to get maximum independent set on original
        # Use Tomita-style branch and bound with greedy coloring upper bound
        # Represent candidate set as bitmask
        # Track best clique (i.e., MIS) mask in outer scope
        best_mask = best_mis_mask  # initialize with greedy MIS

        # Small helpers
        def color_sort(cand_mask: int):
            """
            Greedy coloring to produce an order and upper bounds.
            Returns:
                order: list of vertices in the order they were colored
                bounds: list of color numbers assigned to each vertex in order
            The number of colors assigned up to a vertex serves as an upper bound
            for the size of a clique obtainable from the remaining candidates.
            """
            order: List[int] = []
            bounds: List[int] = []
            rest = cand_mask
            color = 0
            while rest:
                color += 1
                avail = rest
                # Build one color class: a maximal independent set in comp graph's subgraph (i.e., no edges among them)
                while avail:
                    v_bit = avail & -avail
                    v = v_bit.bit_length() - 1
                    order.append(v)
                    bounds.append(color)
                    # Remove v and its neighbors from this color's availability
                    avail &= ~comp_adj[v]
                    avail &= ~v_bit
                    # Remove v from remaining vertices to color
                    rest &= ~v_bit
            return order, bounds

        # Recursive expansion
        # We pass r_mask only when updating best to avoid frequent copying of masks
        def expand(cand_mask: int, r_size: int, r_mask: int):
            nonlocal best_size, best_mask
            if cand_mask == 0:
                if r_size > best_size:
                    best_size = r_size
                    best_mask = r_mask
                return

            order, bounds = color_sort(cand_mask)
            # Iterate vertices in reverse color order
            # Prune when r_size + bounds[i] <= best_size
            for i in range(len(order) - 1, -1, -1):
                if r_size + bounds[i] <= best_size:
                    return
                v = order[i]
                v_bit = 1 << v
                new_cand = cand_mask & comp_adj[v]
                expand(new_cand, r_size + 1, r_mask | v_bit)
                cand_mask &= ~v_bit  # remove v for next iterations

        # Kick off search
        expand(full_mask, 0, 0)

        # best_mask is MIS on original graph (clique on complement)
        # Vertex cover = V \ MIS
        cover_mask = (~best_mask) & full_mask
        cover = []
        m = cover_mask
        while m:
            b = m & -m
            idx = b.bit_length() - 1
            cover.append(idx)
            m &= ~b

        # Return indices sorted (already in increasing order due to lsb scan)
        return cover