import sys
from typing import List

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Exact minimum vertex cover using a simple branch‑and‑bound.
        Works for modest graph sizes (up to ~30 vertices comfortably).
        """
        n = len(problem)
        # build edge list
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if problem[i][j]]
        if not edges:
            return []

        # greedy 2‑approx to obtain an initial upper bound and a concrete cover
        def greedy_cover() -> List[int]:
            remaining = set(edges)
            cover = set()
            while remaining:
                u, v = remaining.pop()
                cover.add(u)
                cover.add(v)
                # remove all edges incident to u or v
                remaining = {e for e in remaining if u not in e and v not in e}
            return list(cover)

        best_solution = greedy_cover()
        best_size = len(best_solution)

        # lower bound: size of a maximal matching (each edge needs a distinct vertex)
        def matching_lower(mask: int) -> int:
            seen = set()
            cnt = 0
            for u, v in edges:
                if (mask >> u) & 1 or (mask >> v) & 1:
                    continue
                if u in seen or v in seen:
                    continue
                seen.add(u)
                seen.add(v)
                cnt += 1
            return cnt

        sys.setrecursionlimit(10000)

        # kernelisation: degree‑1 rule – if a vertex has degree 1, its neighbour must be in the cover
        adj = [0] * n
        for u, v in edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u

        def reduce(mask: int, cnt: int):
            changed = True
            while changed:
                changed = False
                deg = [0] * n
                for u, v in edges:
                    if ((mask >> u) & 1) or ((mask >> v) & 1):
                        continue
                    deg[u] += 1
                    deg[v] += 1
                for v in range(n):
                    if (mask >> v) & 1:
                        continue
                    if deg[v] == 1:
                        # find its unique neighbour
                        for u in range(n):
                            if ((adj[v] >> u) & 1) and not ((mask >> u) & 1):
                                mask |= 1 << u
                                cnt += 1
                                changed = True
                                break
            return mask, cnt

        # depth‑first search with pruning
        def dfs(mask: int, cnt: int):
            nonlocal best_size, best_solution
            if cnt >= best_size:
                return
            lb = matching_lower(mask)
            if cnt + lb >= best_size:
                return
            mask, cnt = reduce(mask, cnt)

            # find an uncovered edge
            for u, v in edges:
                if not ((mask >> u) & 1) and not ((mask >> v) & 1):
                    dfs(mask | (1 << u), cnt + 1)
                    dfs(mask | (1 << v), cnt + 1)
                    return
            # all edges covered – update best solution
            sol = [i for i in range(n) if (mask >> i) & 1]
            if cnt < best_size:
                best_size = cnt
                best_solution = sol

        dfs(0, 0)
        return sorted(best_solution)