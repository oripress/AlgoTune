from __future__ import annotations

from typing import Any, List, Optional, Tuple

class Solver:
    def solve(self, problem, **kwargs) -> Any:  # type: ignore[override]
        """
        Exact graph coloring using a fast DSATUR-based branch-and-bound with component
        decomposition and clique seeding. For each connected component, we binary
        search the minimal K using a K-colorability DSATUR with seeded clique.
        This is typically very fast and exact on moderate graphs.

        Input:
            problem: 2D list adjacency matrix (0/1), symmetric, no self-loops expected.

        Output:
            List[int]: colors 1..k assigned to vertices (k minimal).
        """
        A = problem
        n = len(A)
        if n == 0:
            return []

        # Build adjacency masks and adjacency lists
        adj_mask = [0] * n
        neighbors: List[List[int]] = [[] for _ in range(n)]
        deg = [0] * n
        for i in range(n):
            row = A[i]
            m = 0
            # Creating adjacency mask and neighbor list
            for j, v in enumerate(row):
                if v and i != j:
                    m |= 1 << j
                    neighbors[i].append(j)
            adj_mask[i] = m
            deg[i] = len(neighbors[i])

        # Trivial cases
        if n == 1:
            return [1]
        if all(deg[i] == 0 for i in range(n)):
            # No edges => all vertices can be color 1
            return [1] * n

        # Decompose into connected components
        comps = self._components(neighbors)

        # Solve each component independently, then reuse color labels across components
        colors_global = [0] * n
        for comp in comps:
            if len(comp) == 1:
                colors_global[comp[0]] = 1
                continue

            comp_index = {v: idx for idx, v in enumerate(comp)}
            m = len(comp)

            # Build subgraph structures
            sub_adj_mask = [0] * m
            sub_neighbors: List[List[int]] = [[] for _ in range(m)]
            sub_deg = [0] * m
            for idx, v in enumerate(comp):
                mask = 0
                for u in neighbors[v]:
                    if u in comp_index:
                        j = comp_index[u]
                        mask |= 1 << j
                        sub_neighbors[idx].append(j)
                sub_adj_mask[idx] = mask
                sub_deg[idx] = len(sub_neighbors[idx])

            # Upper bound via greedy largest-first coloring
            greedy_order = sorted(range(m), key=lambda v: (-sub_deg[v], v))
            greedy_colors, ub = self._greedy_coloring(sub_adj_mask, greedy_order)

            # Lower bound via a simple greedy clique heuristic
            clique_vertices = self._greedy_clique_vertices(sub_adj_mask, sub_deg)
            lb = len(clique_vertices)

            # If bounds match, assign greedy coloring
            if lb == ub:
                for idx, v in enumerate(comp):
                    colors_global[v] = greedy_colors[idx] + 1
                continue

            # Binary search K in [lb, ub] using DSATUR with clique seeding
            best_colors: Optional[List[int]] = None
            low, high = lb, ub
            while low < high:
                mid = (low + high) // 2
                feasible, colors_mid = self._k_colorable(
                    sub_adj_mask, sub_neighbors, sub_deg, mid, clique_vertices
                )
                if feasible and colors_mid is not None:
                    high = mid
                    best_colors = colors_mid
                else:
                    low = mid + 1

            k_opt = low
            if best_colors is None or len(set(best_colors)) != k_opt:
                feasible, colors_mid = self._k_colorable(
                    sub_adj_mask, sub_neighbors, sub_deg, k_opt, clique_vertices
                )
                if not feasible or colors_mid is None:
                    # Fallback to greedy (should not happen if logic correct)
                    for idx, v in enumerate(comp):
                        colors_global[v] = greedy_colors[idx] + 1
                    continue
                best_colors = colors_mid

            # Write component result to global colors (temporarily 1..k_comp)
            for idx, v in enumerate(comp):
                colors_global[v] = best_colors[idx] + 1

        # Normalize so colors span 1..k and are contiguous
        used = sorted(set(colors_global))
        remap = {old: new for new, old in enumerate(used, start=1)}
        colors_global = [remap[c] for c in colors_global]
        return colors_global

    @staticmethod
    def _components(neighbors: List[List[int]]) -> List[List[int]]:
        n = len(neighbors)
        seen = [False] * n
        comps: List[List[int]] = []
        for i in range(n):
            if seen[i]:
                continue
            stack = [i]
            seen[i] = True
            comp: List[int] = []
            while stack:
                v = stack.pop()
                comp.append(v)
                for u in neighbors[v]:
                    if not seen[u]:
                        seen[u] = True
                        stack.append(u)
            comps.append(comp)
        return comps

    @staticmethod
    def _greedy_coloring(adj_mask: List[int], order: List[int]) -> Tuple[List[int], int]:
        n = len(adj_mask)
        colors = [-1] * n
        color_masks: List[int] = []  # bitmask of vertices for each color class

        for v in order:
            # Find first color not conflicting with neighbors
            assigned = False
            am = adj_mask[v]
            for c, cm in enumerate(color_masks):
                if (am & cm) == 0:
                    colors[v] = c
                    color_masks[c] |= 1 << v
                    assigned = True
                    break
            if not assigned:
                c_new = len(color_masks)
                colors[v] = c_new
                color_masks.append(1 << v)

        return colors, len(color_masks)

    @staticmethod
    def _greedy_clique_vertices(adj_mask: List[int], deg: List[int]) -> List[int]:
        # Simple greedy clique: iterate vertices by descending degree,
        # add vertex if it connects to all in current clique.
        order = sorted(range(len(adj_mask)), key=lambda v: (-deg[v], v))
        clique: List[int] = []
        clique_mask = 0
        for v in order:
            if (clique_mask & ~adj_mask[v]) == 0:
                clique.append(v)
                clique_mask |= 1 << v
        return clique

    @staticmethod
    def _greedy_clique_lb(adj_mask: List[int], deg: List[int]) -> int:
        # Simple greedy: iterate vertices by descending degree, add if adjacent to all in clique
        order = sorted(range(len(adj_mask)), key=lambda v: (-deg[v], v))
        clique_mask = 0
        for v in order:
            if (clique_mask & ~adj_mask[v]) == 0:
                clique_mask |= 1 << v
        # popcount
        return clique_mask.bit_count()

    def _k_colorable(
        self,
        adj_mask: List[int],
        neighbors: List[List[int]],
        deg: List[int],
        K: int,
        seed_clique: Optional[List[int]] = None,
    ) -> Tuple[bool, Optional[List[int]]]:
        """
        Tests K-colorability using DSATUR backtracking with optional clique seeding.
        Returns (feasible, colors) with colors in 0..K-1 if feasible.
        """
        n = len(adj_mask)
        all_colors_mask = (1 << K) - 1

        # Quick bound: a clique larger than K makes it infeasible
        if seed_clique is not None and len(seed_clique) > K:
            return False, None

        colors = [-1] * n  # assigned colors 0..K-1, -1 unassigned
        sat_mask = [0] * n  # bitmask of colors used by colored neighbors
        sat_deg = [0] * n   # popcount of sat_mask[v]

        # Helpers
        def select_vertex() -> int:
            # choose uncolored vertex with maximum saturation degree; tie -> largest degree; tie -> smallest index
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in range(n):
                if colors[v] != -1:
                    continue
                s = sat_deg[v]
                d = deg[v]
                if s > best_sat or (s == best_sat and d > best_deg):
                    best_sat = s
                    best_deg = d
                    best_v = v
            return best_v

        def assign(v: int, c: int, changed: List[int]) -> bool:
            colors[v] = c
            bit = 1 << c
            for u in neighbors[v]:
                if colors[u] != -1:
                    continue
                if (sat_mask[u] & bit) == 0:
                    sat_mask[u] |= bit
                    sat_deg[u] += 1
                    changed.append(u)
                    if sat_deg[u] >= K:
                        return False
            return True

        def undo(v: int, c: int, changed: List[int]) -> None:
            bit = 1 << c
            for u in changed:
                sat_mask[u] &= ~bit
                sat_deg[u] -= 1
            colors[v] = -1

        # Seed a greedy clique with distinct colors to break symmetry
        seeded_count = 0
        if seed_clique:
            for i, v in enumerate(seed_clique):
                if i >= K:
                    break
                # Check availability for v prior to assignment
                # Since it's a clique, we force distinct colors 0..i
                changed: List[int] = []
                ok = assign(v, i, changed)
                if not ok:
                    # Seeding leads to immediate contradiction
                    return False, None
                seeded_count += 1

        # Pre-check after seeding
        for v in range(n):
            if colors[v] == -1 and sat_deg[v] >= K:
                return False, None

        # Main DFS
        def dfs(remaining: int) -> bool:
            if remaining == 0:
                return True

            v = select_vertex()
            if v == -1:
                return True

            avail_mask = (~sat_mask[v]) & all_colors_mask
            if avail_mask == 0:
                return False

            mlocal = avail_mask
            while mlocal:
                lowest_bit = mlocal & -mlocal
                c = lowest_bit.bit_length() - 1
                mlocal &= mlocal - 1

                changed: List[int] = []
                ok = assign(v, c, changed)
                if ok:
                    if dfs(remaining - 1):
                        return True
                undo(v, c, changed)
            return False

        # Count remaining uncolored after seeding
        remaining = n - seeded_count
        if dfs(remaining):
            return True, colors
        return False, None