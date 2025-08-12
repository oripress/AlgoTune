from __future__ import annotations

from typing import Any, List, Optional

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> List[int]:
        """
        Solve the Traveling Salesman Problem (TSP) given a distance matrix.

        Strategy:
        - For small/medium instances (n <= 16), use Held-Karp dynamic programming (exact, fast in Python for these sizes).
        - For larger instances, use OR-Tools CP-SAT AddCircuit formulation (exact), with a heuristic upper bound and hints.

        :param problem: Distance matrix as list of lists (square, non-negative; 0 on diagonals).
        :return: A list of city indices representing an optimal tour starting and ending at city 0.
        """
        dist = problem
        n = len(dist)

        # Handle trivial cases
        if n <= 1:
            return [0, 0]
        if n == 2:
            return [0, 1, 0]

        # Threshold for Held-Karp DP
        # m = n - 1 (cities excluding 0). Complexity roughly ~ m*(m-1)*2^(m-2) additions.
        # Use a conservative threshold to avoid timeouts when CP-SAT is much faster.
        if n <= 16:
            return self._held_karp(dist)

        # Fallback to CP-SAT circuit (reference style) with UB and hints
        return self._cp_sat(dist)

    def _held_karp(self, dist: List[List[int]]) -> List[int]:
        """
        Held-Karp dynamic programming (exact).
        Uses city 0 as the fixed start/end. Only tracks subsets of cities 1..n-1.
        """
        n = len(dist)
        m = n - 1  # number of "other" cities (1..n-1)

        # Precompute distances among "others" using bit positions [0..m-1] -> original city index (1..n-1)
        # Mapping: bitpos b corresponds to city j = b + 1
        # dd[bk][bj] = dist[city_k][city_j]
        dd = [[0] * m for _ in range(m)]
        for bk in range(m):
            jk = bk + 1
            row_k = dist[jk]
            dd_row = dd[bk]
            for bj in range(m):
                jj = bj + 1
                dd_row[bj] = row_k[jj]

        # Distances from 0 to others and from others to 0
        d0 = [dist[0][j + 1] for j in range(m)]
        d_back = [dist[j + 1][0] for j in range(m)]

        full_mask = (1 << m) - 1

        # dp[mask] is a dict: key = last (bitpos 0..m-1), value = min cost from 0 -> ... -> last visiting exactly 'mask'
        dp: List[Optional[dict[int, int]]] = [None] * (1 << m)

        # Initialize base cases: paths from 0 directly to each city j
        for b in range(m):
            mask = 1 << b
            dp[mask] = {b: d0[b]}

        # Iterative DP over subset sizes using Gosper's hack, processing masks on the fly
        for s in range(2, m + 1):
            mask = (1 << s) - 1
            while mask <= full_mask:
                cur_dict: dict[int, int] = {}
                tmp = mask
                while tmp:
                    lb = tmp & -tmp
                    j = lb.bit_length() - 1  # bit position for 'last' city
                    prev_mask = mask ^ lb
                    prev_dict = dp[prev_mask]
                    if prev_dict:
                        best = None
                        for k, cost_k in prev_dict.items():
                            val = cost_k + dd[k][j]
                            if best is None or val < best:
                                best = val
                        if best is not None:
                            cur_dict[j] = best
                    tmp ^= lb
                dp[mask] = cur_dict
                c = mask & -mask
                r = mask + c
                mask = (((r ^ mask) >> 2) // c) | r

        # Close the tour: choose last city j minimizing dp[full][j] + d_back[j]
        last_dict = dp[full_mask]
        assert last_dict, "DP failed to compute final states"
        best_last = None
        best_total = None
        for j, cost_j in last_dict.items():
            tot = cost_j + d_back[j]
            if best_total is None or tot < best_total or (tot == best_total and (best_last is None or j < best_last)):
                best_total = tot
                best_last = j

        # Reconstruct path by backtracking without storing parents
        path_bits_reversed: List[int] = []
        mask = full_mask
        j = best_last  # current last bitpos
        while mask:
            path_bits_reversed.append(j)
            lb = 1 << j
            prev_mask = mask ^ lb
            if prev_mask == 0:
                break
            prev_dict = dp[prev_mask]
            # Choose predecessor k achieving dp[prev_mask][k] + dd[k][j] == dp[mask][j]
            target = dp[mask][j]
            best_k = None
            best_val = None
            for k, cost_k in prev_dict.items():
                val = cost_k + dd[k][j]
                if val == target:
                    if best_k is None or k < best_k:
                        best_k = k
                        best_val = val
            if best_k is None:
                for k, cost_k in prev_dict.items():
                    val = cost_k + dd[k][j]
                    if best_val is None or val < best_val or (val == best_val and (best_k is None or k < best_k)):
                        best_k = k
                        best_val = val
            j = best_k  # type: ignore[assignment]
            mask = prev_mask

        # Construct the final tour: [0] + reversed(bits) mapped to cities + [0]
        tour = [0]
        for b in reversed(path_bits_reversed):
            tour.append(b + 1)
        tour.append(0)
        return tour

    def _heuristic_tour(self, dist: List[List[int]]) -> List[int]:
        """Construct a quick feasible tour via nearest-neighbor then polish with 2-opt."""
        n = len(dist)
        D = dist
        # Nearest neighbor starting at 0
        visited = [False] * n
        visited[0] = True
        tour = [0]
        cur = 0
        for _ in range(1, n):
            best_j = -1
            best_d = None
            row = D[cur]
            for j in range(n):
                if not visited[j] and j != cur:
                    d = row[j]
                    if best_d is None or d < best_d:
                        best_d = d
                        best_j = j
            if best_j == -1:
                # Fallback (should not happen)
                for j in range(n):
                    if not visited[j]:
                        best_j = j
                        break
            visited[best_j] = True
            tour.append(best_j)
            cur = best_j
        tour.append(0)
        # 2-opt improvement (first-improvement)
        tour = self._two_opt(tour, D)
        return tour

    @staticmethod
    def _two_opt(tour: List[int], D: List[List[int]]) -> List[int]:
        n = len(tour) - 1  # number of unique nodes
        improved = True
        # Limit iterations to avoid pathological cases
        max_passes = 20
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            # i from 1 to n-2, j from i+1 to n-1
            for i in range(1, n - 1):
                a = tour[i - 1]
                b = tour[i]
                dab = D[a][b]
                for j in range(i + 1, n):
                    c = tour[j]
                    d = tour[j + 1]
                    # Skip if edges share a node after wraparound checks
                    if b == c or a == d:
                        continue
                    # delta = (a->c + b->d) - (a->b + c->d)
                    delta = D[a][c] + D[b][d] - (dab + D[c][d])
                    if delta < 0:
                        # Improve by reversing segment [i..j]
                        tour[i:j + 1] = reversed(tour[i:j + 1])
                        improved = True
                        break
                if improved:
                    break
        return tour

    @staticmethod
    def _tour_cost(tour: List[int], D: List[List[int]]) -> int:
        cost = 0
        for i in range(len(tour) - 1):
            cost += D[tour[i]][tour[i + 1]]
        return cost

    def _cp_sat(self, dist: List[List[int]]) -> List[int]:
        """Fallback to CP-SAT AddCircuit formulation (exact), with upper bound and hint."""
        try:
            from ortools.sat.python import cp_model
        except Exception:
            # If OR-Tools is unavailable, fallback to Held-Karp regardless (may be slow).
            return self._held_karp(dist)

        n = len(dist)
        model = cp_model.CpModel()

        # Heuristic UB and hints (compute first to prune variables)
        heuristic_tour = self._heuristic_tour(dist)
        ub = self._tour_cost(heuristic_tour, dist)

        # Create variables for arcs i->j, i != j, but prune arcs with cost > ub (cannot appear in any solution with obj <= ub)
        x = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if dist[i][j] <= ub:
                    x[(i, j)] = model.NewBoolVar(f"x[{i},{j}]")

        # Circuit constraint enforces a single Hamiltonian cycle
        arcs = []
        for (i, j), var in x.items():
            arcs.append((i, j, var))
        model.AddCircuit(arcs)

        # Objective: minimize total cost
        obj = sum(dist[i][j] * x[(i, j)] for (i, j) in x.keys())
        model.Minimize(obj)

        # Constrain objective to be <= UB to reduce search
        model.Add(obj <= ub)

        # Add hints for arcs in heuristic tour (ensure variables exist)
        for i in range(n):
            u = heuristic_tour[i]
            v = heuristic_tour[i + 1]
            if u != v and (u, v) in x:
                model.AddHint(x[(u, v)], 1)

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        # Enable multithreading to speed up proving optimality when available.
        solver.parameters.num_search_workers = 8

        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return []

        # Extract the cycle starting from 0
        tour = []
        current = 0
        visited = 0
        while visited < n:
            tour.append(current)
            found = False
            # Find next city chosen
            for nxt in range(n):
                if current != nxt and (current, nxt) in x and solver.Value(x[(current, nxt)]) == 1:
                    current = nxt
                    found = True
                    break
            if not found:
                break
            visited += 1
        tour.append(0)
        return tour