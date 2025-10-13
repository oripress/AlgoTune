from __future__ import annotations

from typing import Any, List

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Maximum Clique via bitset-based branch-and-bound (Tomita-style with greedy coloring).

        Input:
            problem: 2D 0/1 adjacency matrix (symmetric, no self-loops expected)

        Output:
            List of node indices forming a maximum clique (any one of them).
        """
        # Handle trivial cases
        if problem is None:
            return []
        n = len(problem)
        if n == 0:
            return []
        if n == 1:
            return [0]

        # Build adjacency bitsets in original indexing
        # adj_orig[i]: bitset of neighbors of i in original indexing
        adj_orig = [0] * n
        for i, row in enumerate(problem):
            bits = 0
            # iterate row; treat any non-zero as an edge
            # Avoid RangeError for malformed rows
            m = min(n, len(row))
            for j in range(m):
                if row[j]:
                    bits |= 1 << j
            # remove self-loop if present
            bits &= ~(1 << i)
            adj_orig[i] = bits

        # Compute degrees to derive a static vertex ordering (desc by degree)
        degrees = [adj_orig[i].bit_count() for i in range(n)]
        # Order original vertices by degree descending (break ties by index for determinism)
        perm = sorted(range(n), key=lambda v: (degrees[v], v), reverse=True)
        rank = [0] * n  # original -> new
        for new_idx, orig_idx in enumerate(perm):
            rank[orig_idx] = new_idx

        # Build adjacency in new ordering using bitset remapping via set bits iteration
        adj = [0] * n
        for new_i, orig_i in enumerate(perm):
            b = adj_orig[orig_i]
            new_bits = 0
            while b:
                lsb = b & -b
                j = (lsb.bit_length() - 1)
                new_bits |= 1 << rank[j]
                b ^= lsb
            # ensure no self loop bit set
            new_bits &= ~(1 << new_i)
            adj[new_i] = new_bits

        # Helpers
        ALL = (1 << n) - 1

        # Greedy initial clique (provides a lower bound)
        # Iterate in current (degree-descending) order: vertices 0..n-1
        best_clique: List[int] = []
        common = ALL
        for v in range(n):
            if (common >> v) & 1:
                best_clique.append(v)
                common &= adj[v]

        # Bit operations helpers
        def lsb_index(x: int) -> int:
            # Index of least significant set bit
            return (x & -x).bit_length() - 1

        # Greedy coloring to compute upper bounds and candidate order
        # Returns (order_list, color_bounds) where bounds are nondecreasing
        def color_sort(P: int):
            order: List[int] = []
            bounds: List[int] = []
            U = P
            color = 0
            # Greedy coloring using independent sets within U
            while U:
                color += 1
                Q = U
                while Q:
                    v_bit = Q & -Q
                    v = v_bit.bit_length() - 1
                    order.append(v)
                    bounds.append(color)
                    # remove v from U
                    U &= ~v_bit
                    # remove neighbors of v from current color class
                    Q &= ~adj[v]
                    # keep Q subset of U (uncolored vertices)
                    Q &= U
            return order, bounds

        best_size = len(best_clique)

        # Branch and bound search (Tomita-style)
        def expand(R: List[int], P: int):
            nonlocal best_clique, best_size
            if not P:
                # R is a maximal clique
                if len(R) > best_size:
                    best_clique = R.copy()
                    best_size = len(R)
                return

            order, bounds = color_sort(P)
            # Iterate in reverse order (largest color bound first)
            # Perform pruning using color (upper bound)
            for idx in range(len(order) - 1, -1, -1):
                # Bound: |R| + bounds[idx] <= best_size => cannot improve
                if len(R) + bounds[idx] <= best_size:
                    return
                v = order[idx]
                v_bit = 1 << v
                # Choose v
                R.append(v)
                newP = P & adj[v]
                if newP:
                    expand(R, newP)
                else:
                    # R is a clique; update best
                    if len(R) > best_size:
                        best_clique = R.copy()
                        best_size = len(R)
                R.pop()
                # Exclude v from P for subsequent branches
                P &= ~v_bit
                if not P:
                    return

        # Run the search
        expand([], ALL)

        # Map best clique back to original indices (inverse permutation)
        # best_clique currently in new indexing (0..n-1), which corresponds to orig indices perm[new]
        result = sorted(perm[v] for v in best_clique)
        return result