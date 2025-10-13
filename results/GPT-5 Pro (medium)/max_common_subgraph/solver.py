from __future__ import annotations

from typing import Any, List, Tuple

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        # Parse input
        A = problem["A"]
        B = problem["B"]
        n = len(A)
        m = len(B)
        if n == 0 or m == 0:
            return []

        # Build association graph P with vertices (i, p), edges between (i,p) and (j,q)
        # iff i != j, p != q, and A[i][j] == B[p][q].
        # Then maximum common induced subgraph equals maximum clique in P.

        N = n * m  # number of vertices in association graph

        # Precompute bit masks for B rows: T1[p] bits=1 where B[p][q]==1 and q != p
        # and T0[p] bits=1 where B[p][q]==0 and q != p
        row1_bits_B: List[int] = [0] * m
        row0_bits_B: List[int] = [0] * m
        full_m_mask = (1 << m) - 1
        for p in range(m):
            bits1 = 0
            row_p = B[p]
            # Build bits1 for q != p and B[p][q] == 1
            # Assume symmetry but handle general case
            for q, val in enumerate(row_p):
                if q != p and val:
                    bits1 |= (1 << q)
            row1_bits_B[p] = bits1
            # T0 = all q minus bits1 minus self bit
            row0_bits_B[p] = (full_m_mask ^ bits1) & ~(1 << p)

        # Precompute j lists for A rows: S1[i] = {j != i | A[i][j]==1}, S0[i] similar.
        j1_list: List[List[int]] = [[] for _ in range(n)]
        j0_list: List[List[int]] = [[] for _ in range(n)]
        for i in range(n):
            row_i = A[i]
            j1 = j1_list[i]
            j0 = j0_list[i]
            for j, val in enumerate(row_i):
                if j == i:
                    continue
                if val:
                    j1.append(j)
                else:
                    j0.append(j)

        # Utility to compute neighbors bitset for node u=(i,p)
        # Using block composition: neighbors = union over j in S1[i] { T1[p] << (j*m) } U
        #                                union over j in S0[i] { T0[p] << (j*m) }
        def build_neighbors_for(i: int, p: int) -> int:
            nb = 0
            t1 = row1_bits_B[p]
            t0 = row0_bits_B[p]
            # For 1-edges alignment
            base_mult = m  # shift unit per j
            for j in j1_list[i]:
                nb |= (t1 << (j * base_mult))
            # For 0-edges alignment
            if t0:
                for j in j0_list[i]:
                    nb |= (t0 << (j * base_mult))
            return nb

        # Build adjacency bitsets for P
        adj: List[int] = [0] * N
        # Also store mapping (i,p) for each node index
        idx_to_pair_i: List[int] = [0] * N
        idx_to_pair_p: List[int] = [0] * N
        for i in range(n):
            base_i = i * m
            for p in range(m):
                u = base_i + p
                idx_to_pair_i[u] = i
                idx_to_pair_p[u] = p
                adj[u] = build_neighbors_for(i, p)

        # Maximum clique via bitset-based branch and bound (Tomita-style with greedy coloring)
        best_size = 0
        best_clique_mask = 0

        # Precompute max possible size (cannot exceed min(n, m))
        max_possible = min(n, m)

        # Helpers for bit operations
        def popcount(x: int) -> int:
            return x.bit_count()

        def lsb_index(x: int) -> int:
            # Returns index of least significant set bit in x (x != 0)
            return (x & -x).bit_length() - 1

        # Greedy sequential coloring to produce order and color bounds
        # Returns (order_list, color_list) where colors are nondecreasing along order
        def color_sort(cand: int):
            order: List[int] = []
            colors: List[int] = []
            uncolored = cand
            c = 0
            while uncolored:
                c += 1
                # Candidates for current color: vertices not yet colored and
                # not adjacent to any vertex already colored with this color
                # We'll greedily pick vertices for this color
                # allowed initially all uncolored
                allowed = uncolored
                # For performance, we iteratively remove neighbors of chosen vertices
                while allowed:
                    v_bit = allowed & -allowed
                    v = v_bit.bit_length() - 1
                    order.append(v)
                    colors.append(c)
                    # Remove v from uncolored
                    uncolored &= ~v_bit
                    # For this color, remove neighbors of v from allowed
                    allowed &= ~adj[v]
                    # Also remove v itself (already removed via uncolored line)
                    allowed &= ~v_bit
            return order, colors

        # Initial lower bound via greedy heuristic using coloring order
        def initial_heuristic_clique() -> int:
            nonlocal best_clique_mask, best_size
            C = (1 << N) - 1
            order, _ = color_sort(C)
            clique_mask = 0
            size = 0
            # Build clique by scanning in reverse order (higher colors first)
            for v in reversed(order):
                # check if v is adjacent to all in current clique
                if (clique_mask == 0) or ((adj[v] & clique_mask) == clique_mask):
                    clique_mask |= (1 << v)
                    size += 1
                    if size == max_possible:
                        break
            best_clique_mask = clique_mask
            best_size = size
            return size

        # Run initial heuristic
        initial_heuristic_clique()
        if best_size == max_possible:
            # Early exit, already optimal
            res: List[Tuple[int, int]] = []
            mask = best_clique_mask
            while mask:
                v = lsb_index(mask)
                mask &= mask - 1
                res.append((idx_to_pair_i[v], idx_to_pair_p[v]))
            return res

        # Recursive expand
        current_clique_mask = 0

        # We'll implement recursion using stack to minimize Python recursion overhead somewhat,
        # but a recursive function is clearer and performant enough for expected sizes.
        def expand(cand: int):
            nonlocal best_size, best_clique_mask, current_clique_mask
            # Prune by trivial upper bound
            # Use coloring bound
            order, colors = color_sort(cand)
            # Iterate vertices in reverse order (largest color first)
            for idx in range(len(order) - 1, -1, -1):
                v = order[idx]
                color_bound = colors[idx]
                # If even taking the top "color_bound" vertices we can't beat best, prune
                if (popcount(current_clique_mask) + color_bound) <= best_size:
                    return
                v_bit = 1 << v
                # Include v in clique
                current_clique_mask |= v_bit
                new_cand = cand & adj[v]
                if new_cand == 0:
                    # Maximal clique found
                    curr_size = popcount(current_clique_mask)
                    if curr_size > best_size:
                        best_size = curr_size
                        best_clique_mask = current_clique_mask
                        if best_size == max_possible:
                            # Early global stop condition
                            current_clique_mask &= ~v_bit
                            return
                else:
                    expand(new_cand)
                    if best_size == max_possible:
                        current_clique_mask &= ~v_bit
                        return
                # Backtrack: remove v
                current_clique_mask &= ~v_bit
                # Remove v from candidates for this level
                cand &= ~v_bit

        # Start search with all vertices as candidates
        all_vertices = (1 << N) - 1
        expand(all_vertices)

        # Extract solution pairs from best_clique_mask
        res: List[Tuple[int, int]] = []
        mask = best_clique_mask
        while mask:
            v = (mask & -mask)
            idx = v.bit_length() - 1
            mask ^= v
            res.append((idx_to_pair_i[idx], idx_to_pair_p[idx]))

        return res