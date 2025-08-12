from __future__ import annotations
from typing import List

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> List[int]:
        """
        Exact minimum graph coloring using a DSATUR‑based branch‑and‑bound algorithm.

        Parameters
        ----------
        problem : List[List[int]]
            Symmetric adjacency matrix of the undirected graph.

        Returns
        -------
        List[int]
            Color assignment for each vertex, using consecutive colors starting at 1.
        """
        n = len(problem)
        if n == 0:
            return []

        # Build adjacency list
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j]:
                    adj[i].add(j)
                    adj[j].add(i)

        # Trivial case: no edges
        if all(len(nei) == 0 for nei in adj):
            return [1] * n

        # ---------- Greedy upper bound (largest‑first) ----------
        order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
        greedy_colors = [0] * n
        used = 0
        for v in order:
            forbidden = {greedy_colors[u] for u in adj[v] if greedy_colors[u] != 0}
            c = 1
            while c in forbidden:
                c += 1
            greedy_colors[v] = c
            used = max(used, c)

        best_num = used
        best_assign = greedy_colors[:]

        # ---------- DSATUR exact search ----------
        colors = [0] * n                     # current color assignment
        sat_deg = [0] * n                    # saturation degree of each vertex
        neigh_color_sets = [set() for _ in range(n)]

        def select_vertex() -> int:
            """Select uncolored vertex with highest saturation (break ties by degree)."""
            cand = -1
            max_sat = -1
            max_deg = -1
            for v in range(n):
                if colors[v] == 0:
                    if sat_deg[v] > max_sat or (sat_deg[v] == max_sat and len(adj[v]) > max_deg):
                        cand = v
                        max_sat = sat_deg[v]
                        max_deg = len(adj[v])
            return cand

        def backtrack(current_max: int) -> None:
            """Recursive branch‑and‑bound search."""
            nonlocal best_num, best_assign
            # Prune if already using too many colors
            if current_max >= best_num:
                return

            # All vertices colored → update best solution
            if all(c != 0 for c in colors):
                best_num = current_max
                best_assign = colors[:]
                return

            v = select_vertex()
            # Try existing colors first
            for c in range(1, current_max + 1):
                conflict = False
                for u in adj[v]:
                    if colors[u] == c:
                        conflict = True
                        break
                if conflict:
                    continue
                colors[v] = c
                changed = []
                for u in adj[v]:
                    if colors[u] == 0 and c not in neigh_color_sets[u]:
                        neigh_color_sets[u].add(c)
                        sat_deg[u] += 1
                        changed.append(u)
                backtrack(current_max)
                for u in changed:
                    neigh_color_sets[u].remove(c)
                    sat_deg[u] -= 1
                colors[v] = 0

            # Try a new color
            new_c = current_max + 1
            if new_c < best_num:
                colors[v] = new_c
                changed = []
                for u in adj[v]:
                    if colors[u] == 0 and new_c not in neigh_color_sets[u]:
                        neigh_color_sets[u].add(new_c)
                        sat_deg[u] += 1
                        changed.append(u)
                backtrack(new_c)
                for u in changed:
                    neigh_color_sets[u].remove(new_c)
                    sat_deg[u] -= 1
                colors[v] = 0

        # Start the exact search
        backtrack(0)

        # Normalize colors to be consecutive starting at 1
        used_colors = sorted(set(best_assign))
        remap = {old: new for new, old in enumerate(used_colors, start=1)}
        result = [remap[c] for c in best_assign]

        return result