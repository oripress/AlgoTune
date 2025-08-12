from typing import Any, List
import math
import random

class Solver:
    def solve(self, problem: List[List[float]], **kwargs) -> Any:
        """
        Solve the Traveling Salesman Problem.

        - For small n (<= MAX_EXACT_N) use Held-Karp exact DP.
        - For larger n use nearest-neighbor greedy plus 2-opt local search.

        Returns a tour (list of city indices) starting and ending at 0 with length n+1.
        """
        n = len(problem)
        if n <= 0:
            return []
        if n == 1:
            return [0, 0]
        if n == 2:
            return [0, 1, 0]

        dist = problem  # distance matrix

        # Make behavior deterministic across runs (seed RNG)
        random.seed(0)

        # Allow override of exact threshold for testing
        MAX_EXACT_N = int(kwargs.get("max_exact_n", 20))
        if n <= MAX_EXACT_N:
            m = n - 1  # we index cities 1..n-1 by 0..m-1 in the mask
            maxmask = 1 << m
            INF = float("inf")

            dp: List[List[float]] = [[INF] * m for _ in range(maxmask)]
            parent: List[List[int]] = [[-1] * m for _ in range(maxmask)]

            # initialize single-city subsets
            for i in range(m):
                mask = 1 << i
                dp[mask][i] = dist[0][i + 1]
                parent[mask][i] = -1

            # fill DP
            for mask in range(1, maxmask):
                # skip singletons (already initialized)
                if mask & (mask - 1) == 0:
                    continue
                mm = mask
                while mm:
                    lb = mm & -mm
                    j = lb.bit_length() - 1
                    mm -= lb
                    prev_mask = mask ^ (1 << j)
                    best_cost = INF
                    best_prev = -1
                    pm = prev_mask
                    # iterate previous cities k in prev_mask
                    while pm:
                        pb = pm & -pm
                        k = pb.bit_length() - 1
                        pm -= pb
                        val = dp[prev_mask][k]
                        if val == INF:
                            continue
                        cost = val + dist[k + 1][j + 1]
                        if cost < best_cost:
                            best_cost = cost
                            best_prev = k
                    dp[mask][j] = best_cost
                    parent[mask][j] = best_prev

            fullmask = maxmask - 1
            best = INF
            last = -1
            for i in range(m):
                val = dp[fullmask][i]
                if val == INF:
                    continue
                total_cost = val + dist[i + 1][0]
                if total_cost < best:
                    best = total_cost
                    last = i

            # fallback if something went wrong
            if last == -1:
                return [i for i in range(n)] + [0]

            # reconstruct path
            seq: List[int] = []
            mask = fullmask
            cur = last
            while cur != -1:
                seq.append(cur + 1)
                prev = parent[mask][cur]
                mask ^= 1 << cur
                cur = prev
            seq.reverse()
            tour = [0] + seq + [0]
            return tour

        # Heuristic for larger instances

        def tour_length(t: List[int]) -> float:
            s = 0.0
            for a, b in zip(t, t[1:]):
                s += dist[a][b]
            return s

        def greedy(start: int = 0) -> List[int]:
            visited = [False] * n
            tour = [start]
            visited[start] = True
            cur = start
            for _ in range(n - 1):
                nxt = -1
                bestd = math.inf
                for j in range(n):
                    if not visited[j]:
                        d = dist[cur][j]
                        if d < bestd:
                            bestd = d
                            nxt = j
                if nxt == -1:
                    break
                tour.append(nxt)
                visited[nxt] = True
                cur = nxt
            tour.append(start)
            return tour

        def rotate_to_zero(t: List[int]) -> List[int]:
            # rotate cycle so it starts and ends at 0
            if not t:
                return t
            cycle = t[:-1]
            if 0 not in cycle:
                # fallback: create trivial tour
                return [0] + cycle + [0]
            pos = cycle.index(0)
            newcycle = cycle[pos:] + cycle[:pos]
            return newcycle + [newcycle[0]]

        def two_opt(route: List[int], it_limit: int = 50) -> List[int]:
            r = route[:]
            n_t = len(r)
            if n_t <= 3:
                return r
            improved = True
            it = 0
            tol = 1e-12
            while improved and it < it_limit:
                improved = False
                it += 1
                for i in range(1, n_t - 2):
                    ai = r[i - 1]
                    bi = r[i]
                    for j in range(i + 1, n_t - 1):
                        cj = r[j]
                        dj = r[j + 1]
                        delta = (dist[ai][cj] + dist[bi][dj]) - (dist[ai][bi] + dist[cj][dj])
                        if delta < -tol:
                            # reverse segment i..j
                            r[i:j + 1] = r[i:j + 1][::-1]
                            improved = True
                            break
                    if improved:
                        break
            return r

        # initial solution
        best_tour = rotate_to_zero(greedy(0))
        best_tour = two_opt(best_tour, it_limit=30)
        best_len = tour_length(best_tour)

        # random restarts with different greedy starts
        tries = min(10, n)
        starts = list(range(1, min(n, tries)))
        random.shuffle(starts)
        for s in starts:
            t = rotate_to_zero(greedy(s))
            t = two_opt(t, it_limit=15)
            l = tour_length(t)
            if l < best_len:
                best_len = l
                best_tour = t

        # final safety: ensure proper format
        if len(best_tour) != n + 1 or best_tour[0] != 0 or best_tour[-1] != 0:
            return [i for i in range(n)] + [0]

        return best_tour