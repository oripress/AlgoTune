from __future__ import annotations

from array import array
from typing import Any, List

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> List[int]:
        """
        Solve TSP optimally. Uses Held-Karp DP for small/medium n for speed, and
        falls back to OR-Tools CP-SAT for larger n.

        :param problem: Distance matrix (square), positive off-diagonals, 0 on diagonal.
        :return: Tour as list of city indices, starts and ends at 0, length n+1.
        """
        A = problem
        n = len(A)

        # Handle trivial cases
        if n == 0:
            return []
        if n == 1:
            return [0, 0]
        if n == 2:
            return [0, 1, 0]

        # Held-Karp DP threshold: m = n-1
        m = n - 1
        # Choose a safe threshold to avoid excessive memory usage
        # dp size ~ m * 2^m doubles; at m=18 it's ~36MB, which is acceptable.
        if m <= 18:
            return self._solve_dp(A)

        # Fallback to OR-Tools CP-SAT for larger instances
        try:
            from ortools.sat.python import cp_model

            model = cp_model.CpModel()
            # Create variables for directed arcs i != j
            x = {(i, j): model.NewBoolVar(f"x[{i},{j}]")
                 for i in range(n) for j in range(n) if i != j}

            # Circuit constraint: exactly one Hamiltonian cycle visiting all nodes
            model.AddCircuit([(i, j, var) for (i, j), var in x.items()])

            # Objective: minimize total distance
            model.Minimize(sum(A[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j))

            solver = cp_model.CpSolver()
            # Be quiet and rely on default time limits
            solver.parameters.log_search_progress = False
            status = solver.Solve(model)

            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                path = []
                current = 0
                visited_count = 0
                while visited_count < n:
                    path.append(current)
                    visited_count += 1
                    nxt = None
                    for j in range(n):
                        if current != j and solver.Value(x[current, j]) == 1:
                            nxt = j
                            break
                    if nxt is None:
                        break
                    current = nxt
                path.append(0)
                return path
            else:
                # As a last resort, return a simple valid tour (greedy), though optimality may fail.
                return self._greedy_tour(A)
        except Exception:
            # If OR-Tools is not available or fails, return greedy as a fallback.
            return self._greedy_tour(A)

    @staticmethod
    def _greedy_tour(A: List[List[int]]) -> List[int]:
        n = len(A)
        if n == 0:
            return []
        tour = [0]
        used = [False] * n
        used[0] = True
        cur = 0
        for _ in range(n - 1):
            best = None
            best_d = float("inf")
            for j in range(n):
                if not used[j] and j != cur:
                    d = A[cur][j]
                    if d < best_d:
                        best_d = d
                        best = j
            if best is None:
                # fallback safety
                for j in range(1, n):
                    if not used[j]:
                        best = j
                        break
            used[best] = True
            tour.append(best)
            cur = best
        tour.append(0)
        return tour

    @staticmethod
    def _solve_dp(A: List[List[int]]) -> List[int]:
        # Held-Karp DP with flat array storage for speed
        n = len(A)
        m = n - 1  # number of non-start nodes (1..n-1 mapped to positions 0..m-1)
        maxmask = 1 << m
        inf = float("inf")

        # Flat dp array of size maxmask * m, storing only states where j in mask
        # dp[mask, jpos] = min cost from 0 -> ... -> (mask) -> j
        dp = array("d", [inf]) * (maxmask * m)

        # Base cases: masks with single bit jpos
        row0 = A[0]
        # map position to city index: pos 0 -> city 1, pos 1 -> city 2, ..., pos m-1 -> city m
        # Initialize dp for singletons
        for jpos in range(m):
            mask = 1 << jpos
            dp[mask * m + jpos] = float(row0[jpos + 1])

        # Transition
        for mask in range(1, maxmask):
            sub = mask
            while sub:
                lowbit = sub & -sub
                jpos = (lowbit.bit_length() - 1)
                pmask = mask ^ lowbit
                if pmask != 0:
                    best = inf
                    inner = pmask
                    jcity = jpos + 1
                    while inner:
                        l2 = inner & -inner
                        kpos = (l2.bit_length() - 1)
                        # dp[pmask, kpos] + A[kcity][jcity]
                        val = dp[pmask * m + kpos] + A[kpos + 1][jcity]
                        if val < best:
                            best = val
                        inner ^= l2
                    dp[mask * m + jpos] = best
                sub ^= lowbit

        # Close the tour: choose end j that minimizes dp[full, j] + A[j][0]
        full = maxmask - 1
        best_cost = inf
        best_end_pos = -1
        for jpos in range(m):
            val = dp[full * m + jpos] + A[jpos + 1][0]
            if val < best_cost:
                best_cost = val
                best_end_pos = jpos

        # Reconstruct path from best_end_pos by dynamic backtracking without parent table
        path_positions = [best_end_pos]
        mask = full
        last_pos = best_end_pos
        while mask != (1 << last_pos):
            pmask = mask ^ (1 << last_pos)
            best_prev = -1
            best_val = inf
            inner = pmask
            jcity = last_pos + 1
            while inner:
                l2 = inner & -inner
                kpos = (l2.bit_length() - 1)
                val = dp[pmask * m + kpos] + A[kpos + 1][jcity]
                if val < best_val:
                    best_val = val
                    best_prev = kpos
                inner ^= l2
            path_positions.append(best_prev)
            last_pos = best_prev
            mask = pmask

        # Convert positions back to cities and order from start to end
        path_positions.reverse()
        tour = [0] + [p + 1 for p in path_positions] + [0]
        return tour