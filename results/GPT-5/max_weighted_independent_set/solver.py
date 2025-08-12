from __future__ import annotations

from typing import Any, Dict, List, Tuple

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the Maximum Weighted Independent Set problem.

        Approach:
        - Filter out non-positive weight vertices (they never help optimal MWIS).
        - Decompose into connected components (on the induced subgraph of remaining vertices).
        - For each component, solve exactly via branch-and-bound Maximum Weight Clique
          on the complement graph with a greedy weighted coloring upper bound (bitset implementation).
        - Include very cheap early checks for trivial edgeless and clique components.
        - Combine component solutions and map back to original indices.

        Input:
            problem: dict with keys:
                - 'adj_matrix': 2D list (n x n) with 0/1 entries
                - 'weights': list of length n with node weights

        Output:
            List of selected node indices (independent set).
        """
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]

        n = len(adj_matrix)
        if n == 0:
            return []

        if len(weights) != n:
            raise ValueError("weights length must match adj_matrix size")

        # Keep only strictly positive weight vertices
        positive = [i for i, w in enumerate(weights) if (w is not None and w > 0)]
        if not positive:
            return []

        # Precompute neighbors among positive vertices for connected components and masks
        pos_neighbors: Dict[int, List[int]] = {}
        deg_pos: Dict[int, int] = {}
        for u in positive:
            row = adj_matrix[u]
            nbs = [v for v in positive if row[v]]
            pos_neighbors[u] = nbs
            deg_pos[u] = len(nbs)

        # Build connected components on the induced subgraph of 'positive' vertices
        comps: List[List[int]] = []
        remaining = set(positive)
        while remaining:
            v0 = remaining.pop()
            stack = [v0]
            comp = [v0]
            while stack:
                u = stack.pop()
                for w in pos_neighbors[u]:
                    if w in remaining:
                        remaining.remove(w)
                        stack.append(w)
                        comp.append(w)
            comps.append(comp)

        # Helper: bit operations
        def lsb_index(x: int) -> int:
            return (x & -x).bit_length() - 1

        # Branch-and-bound MW clique in complement (exact per component)
        def solve_component_bnb(comp_raw: List[int]) -> List[int]:
            # Reorder vertices by descending weight, tie by ascending degree to improve branching
            comp = sorted(comp_raw, key=lambda v: (-weights[v], deg_pos[v]))
            k = len(comp)
            if k == 0:
                return []
            if k == 1:
                return [comp[0]]
            if k == 2:
                u0, v0 = comp[0], comp[1]
                if adj_matrix[u0][v0]:
                    return [u0] if weights[u0] >= weights[v0] else [v0]
                else:
                    return [u0, v0]

            # Local weights
            w_loc = [weights[v] for v in comp]

            # Original adjacency masks within component using precomputed pos_neighbors
            idx_of = {v: i for i, v in enumerate(comp)}
            n_orig_masks = [0] * k
            for i, u in enumerate(comp):
                mask = 0
                for nb in pos_neighbors[u]:
                    j = idx_of.get(nb)
                    if j is not None:
                        mask |= (1 << j)
                mask &= ~(1 << i)  # clear self just in case
                n_orig_masks[i] = mask

            # Quick structure checks on original induced subgraph
            deg_list = [n_orig_masks[i].bit_count() for i in range(k)]
            if all(d == 0 for d in deg_list):
                # edgeless: whole set is an independent set
                return comp.copy()
            if all(d == k - 1 for d in deg_list):
                # clique: pick the heaviest vertex
                max_i = max(range(k), key=lambda i: w_loc[i])
                return [comp[max_i]]

            # Complement adjacency within component (exclude self)
            full_mask = (1 << k) - 1
            n_comp_masks = [0] * k
            n_comp_masks_incl = [0] * k
            for i in range(k):
                comp_neigh = (full_mask ^ n_orig_masks[i]) & ~(1 << i)
                n_comp_masks[i] = comp_neigh
                n_comp_masks_incl[i] = comp_neigh | (1 << i)

            # Initial incumbent via two greedy strategies

            # 1) Greedy independent set on original graph using weight/degree ratio
            def greedy_init_is_original() -> Tuple[int, int]:
                total = 0
                mask_sel = 0
                order_loc = list(range(k))
                order_loc.sort(key=lambda x: (w_loc[x] / (deg_list[x] + 1), w_loc[x]), reverse=True)
                alive_mask = full_mask
                for v in order_loc:
                    if (alive_mask >> v) & 1:
                        mask_sel |= (1 << v)
                        total += w_loc[v]
                        alive_mask &= ~(n_orig_masks[v] | (1 << v))
                return total, mask_sel

            # 2) Greedy clique in complement: pick heaviest feasible iteratively
            def greedy_init_clique_complement() -> Tuple[int, int]:
                P = full_mask
                total = 0
                mask_sel = 0
                wl = w_loc
                ncm = n_comp_masks
                while P:
                    x = P
                    max_w = -1
                    v_best = -1
                    while x:
                        v = lsb_index(x)
                        wv = wl[v]
                        if wv > max_w:
                            max_w = wv
                            v_best = v
                        x &= x - 1
                    mask_sel |= (1 << v_best)
                    total += wl[v_best]
                    P &= ncm[v_best]
                return total, mask_sel

            best_w1, best_mask1 = greedy_init_is_original()
            best_w2, best_mask2 = greedy_init_clique_complement()
            if best_w2 > best_w1:
                best_w = best_w2
                best_mask = best_mask2
            else:
                best_w = best_w1
                best_mask = best_mask1

            # Greedy weighted coloring to produce order and bounds for pruning (fast LSB variant)
            def color_order(P_mask: int) -> Tuple[List[int], List[int]]:
                order: List[int] = []
                bounds: List[int] = []
                P_rem = P_mask
                sum_ub = 0
                nci = n_comp_masks_incl
                wl = w_loc
                while P_rem:
                    Q = P_rem
                    color_mask = 0
                    wmax_c = 0
                    while Q:
                        v = lsb_index(Q)
                        color_mask |= (1 << v)
                        wv = wl[v]
                        if wv > wmax_c:
                            wmax_c = wv
                        # remove v and its comp-neighbors from candidate set for this color
                        Q &= ~nci[v]
                    sum_ub += wmax_c
                    cm = color_mask
                    while cm:
                        v = lsb_index(cm)
                        order.append(v)
                        bounds.append(sum_ub)
                        cm &= cm - 1
                    P_rem &= ~color_mask
                return order, bounds

            # Main recursion using bit masks for current clique to avoid list copying
            def expand(P_mask: int, cw: int, curr_mask: int) -> Tuple[int, int]:
                nonlocal best_w, best_mask
                if not P_mask:
                    if cw > best_w:
                        best_w = cw
                        best_mask = curr_mask
                    return best_w, best_mask

                order, ubounds = color_order(P_mask)
                P_local = P_mask
                # Iterate vertices in reverse order (Tomita-style)
                for idx in range(len(order) - 1, -1, -1):
                    # Prune by bound
                    if cw + ubounds[idx] <= best_w:
                        return best_w, best_mask
                    v = order[idx]
                    if (P_local >> v) & 1 == 0:
                        continue
                    cw2 = cw + w_loc[v]
                    new_mask = curr_mask | (1 << v)
                    # Update incumbent
                    if cw2 > best_w:
                        best_w = cw2
                        best_mask = new_mask
                    # Next candidate set: vertices adjacent (in complement) to v and still in P_local
                    P_next = P_local & n_comp_masks[v]
                    if P_next:
                        best_w, best_mask = expand(P_next, cw2, new_mask)
                    # Remove v from candidates before moving to next
                    P_local &= ~(1 << v)
                return best_w, best_mask

            expand(full_mask, 0, 0)

            # Map back to original indices from best_mask
            res: List[int] = []
            bm = best_mask
            while bm:
                v = lsb_index(bm)
                res.append(comp[v])
                bm &= bm - 1
            return res

        # Solve per component
        solution: List[int] = []
        for comp in comps:
            if not comp:
                continue
            if len(comp) == 1:
                solution.append(comp[0])
            else:
                chosen = solve_component_bnb(comp)
                solution.extend(chosen)

        solution.sort()
        return solution