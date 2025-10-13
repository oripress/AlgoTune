from __future__ import annotations

from typing import Any, List, Tuple


class Solver:
    def solve(self, problem, **kwargs) -> Any:  # noqa: ANN401
        """
        Maximum Weighted Independent Set via Maximum Weighted Clique in the complement graph.

        Uses a branch-and-bound with greedy coloring to compute tight upper bounds (weighted variant).
        Representation leverages Python integer bitsets for very fast set operations.

        :param problem: dict with 'adj_matrix' and 'weights'
        :return: list of selected node indices (independent set in the original graph)
        """
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]

        n = len(adj_matrix)
        if n == 0:
            return []

        # Filter out non-positive weight vertices; they can never improve an optimal solution.
        pos_indices = [i for i, w in enumerate(weights) if w > 0]
        if not pos_indices:
            return []

        idx_map = {old: new for new, old in enumerate(pos_indices)}
        rev_map = pos_indices
        m = len(pos_indices)

        # Build complement graph adjacency bitsets among positive-weight vertices.
        # In complement graph Gc: edge exists iff no edge in original and i != j.
        # adjc[i] is a bitmask of neighbors of i in complement graph (in new indexing [0..m-1]).
        adjc = [0] * m
        W = [0] * m
        for new_i, old_i in enumerate(pos_indices):
            W[new_i] = weights[old_i]

        for new_i, old_i in enumerate(pos_indices):
            row = adj_matrix[old_i]
            mask = 0
            # Build adjacency in complement among the reduced set
            for new_j, old_j in enumerate(pos_indices):
                if new_i == new_j:
                    continue
                if row[old_j] == 0:
                    mask |= 1 << new_j
            adjc[new_i] = mask

        # Precompute degrees in complement for possible heuristics (unused but available)
        # degc = [adjc[i].bit_count() for i in range(m)]

        # Greedy initial clique to set a strong initial lower bound (weight).
        all_mask = (1 << m) - 1

        def argmax_weight(mask: int) -> int:
            # Return index with maximal weight in mask; -1 if mask == 0
            if mask == 0:
                return -1
            # Simple scan over set bits
            best_i = -1
            best_w = -1
            x = mask
            while x:
                lsb = x & -x
                i = (lsb.bit_length() - 1)
                wi = W[i]
                if wi > best_w:
                    best_w = wi
                    best_i = i
                x ^= lsb
            return best_i

        # Heuristic: greedy heaviest-first clique in complement
        def greedy_initial_clique(mask: int) -> Tuple[int, int]:
            c_mask = 0
            c_weight = 0
            P = mask
            while P:
                v = argmax_weight(P)
                if v < 0:
                    break
                c_mask |= 1 << v
                c_weight += W[v]
                P &= adjc[v]
            return c_mask, c_weight

        best_mask, best_weight = greedy_initial_clique(all_mask)

        # Coloring-based upper bound (weighted)
        # Return order list and bound array corresponding to prefix sums per color of max weight per color class.
        def color_order_and_bounds(P: int) -> Tuple[List[int], List[int]]:
            # Build color classes as independent sets in the clique graph (Gc) using greedy method
            classes: List[int] = []
            wmaxs: List[int] = []

            remaining = P
            while remaining:
                color_mask = 0
                avail = remaining
                wmax = 0
                # Greedily pick a set of mutually non-adjacent vertices (an independent set in Gc)
                while avail:
                    lsb = avail & -avail
                    v = (lsb.bit_length() - 1)
                    color_mask |= 1 << v
                    wv = W[v]
                    if wv > wmax:
                        wmax = wv
                    # remove v and its neighbors from avail
                    avail &= ~(adjc[v] | (1 << v))
                classes.append(color_mask)
                wmaxs.append(wmax)
                remaining &= ~color_mask

            # Prefix sums of max weights per color
            prefix: List[int] = []
            s = 0
            for wm in wmaxs:
                s += wm
                prefix.append(s)

            order: List[int] = []
            bounds: List[int] = []
            for k, mask in enumerate(classes):
                x = mask
                while x:
                    lsb = x & -x
                    v = (lsb.bit_length() - 1)
                    order.append(v)
                    bounds.append(prefix[k])
                    x ^= lsb
            return order, bounds

        # Branch and Bound
        def expand(P: int, cur_w: int, cur_mask: int) -> None:
            nonlocal best_mask, best_weight
            if P == 0:
                if cur_w > best_weight:
                    best_weight = cur_w
                    best_mask = cur_mask
                return

            order, bounds = color_order_and_bounds(P)
            # Iterate in reverse order as per Tomita-style
            for idx in range(len(order) - 1, -1, -1):
                # If upper bound cannot beat best, prune this node (and all earlier vertices)
                if cur_w + bounds[idx] <= best_weight:
                    return
                v = order[idx]
                # Include v
                new_mask = cur_mask | (1 << v)
                new_w = cur_w + W[v]
                if new_w > best_weight:
                    best_weight = new_w
                    best_mask = new_mask
                P_v = P & adjc[v]
                expand(P_v, new_w, new_mask)
                # Exclude v and continue
                P &= ~(1 << v)

        expand(all_mask, 0, 0)

        # Map back to original graph indices
        result: List[int] = []
        x = best_mask
        while x:
            lsb = x & -x
            v = (lsb.bit_length() - 1)
            result.append(rev_map[v])
            x ^= lsb
        # The independent set can be in any order; sort for determinism
        result.sort()
        return result