class Solver:
    def solve(self, problem, **kwargs):
        import sys
        sys.setrecursionlimit(1000000)
        n = len(problem)
        if n <= 1:
            return [1] * n
        # build adjacency lists and degrees
        neighbors = [[] for _ in range(n)]
        for i, row in enumerate(problem):
            for j, v in enumerate(row):
                if v:
                    neighbors[i].append(j)
        degrees = [len(neighbors[i]) for i in range(n)]
        # greedy Welsh-Powell for upper bound
        order = sorted(range(n), key=lambda x: -degrees[x])
        greedy = [-1] * n
        for v in order:
            used = 0
            for u in neighbors[v]:
                c0 = greedy[u]
                if c0 != -1:
                    used |= 1 << c0
            c = 0
            while (used >> c) & 1:
                c += 1
            greedy[v] = c
        UB = max(greedy) + 1
        # heuristic clique for lower bound
        v0 = max(range(n), key=lambda x: degrees[x])
        clique = [v0]
        cand = [u for u in neighbors[v0]]
        while cand:
            u = max(cand, key=lambda x: degrees[x])
            clique.append(u)
            cand = [w for w in cand if w in neighbors[u]]
        LB = len(clique)
        # if greedy is optimal, return it
        if LB == UB:
            return [c + 1 for c in greedy]
        # branch and bound DSATUR
        best_k = UB
        best_colors = greedy.copy()
        colors = [-1] * n
        neighbor_colors_mask = [0] * n
        # backtracking search
        def backtrack(colored_count, current_k):
            nonlocal best_k, best_colors
            # prune if already no better
            if current_k >= best_k:
                return
            # if all colored, update best
            if colored_count == n:
                best_k = current_k
                best_colors = colors.copy()
                return
            # select next vertex by DSATUR rule
            best_sat = -1
            best_deg = -1
            v_sel = -1
            for i in range(n):
                if colors[i] == -1:
                    sat = neighbor_colors_mask[i].bit_count()
                    deg = degrees[i]
                    if sat > best_sat or (sat == best_sat and deg > best_deg):
                        best_sat = sat
                        best_deg = deg
                        v_sel = i
            used_mask = neighbor_colors_mask[v_sel]
            # try existing colors
            for c in range(current_k):
                if not ((used_mask >> c) & 1):
                    colors[v_sel] = c
                    updated = []
                    for u in neighbors[v_sel]:
                        if colors[u] == -1 and not ((neighbor_colors_mask[u] >> c) & 1):
                            neighbor_colors_mask[u] |= (1 << c)
                            updated.append(u)
                    backtrack(colored_count + 1, current_k)
                    for u in updated:
                        neighbor_colors_mask[u] ^= (1 << c)
                    colors[v_sel] = -1
            # try new color
            if current_k + 1 < best_k:
                c = current_k
                colors[v_sel] = c
                updated = []
                for u in neighbors[v_sel]:
                    if colors[u] == -1 and not ((neighbor_colors_mask[u] >> c) & 1):
                        neighbor_colors_mask[u] |= (1 << c)
                        updated.append(u)
                backtrack(colored_count + 1, current_k + 1)
                for u in updated:
                    neighbor_colors_mask[u] ^= (1 << c)
                colors[v_sel] = -1
        backtrack(0, 0)
        # return 1-based coloring
        return [c + 1 for c in best_colors]