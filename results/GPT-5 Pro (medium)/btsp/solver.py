from __future__ import annotations

from typing import Any, Dict, List, Tuple

import os

from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the Bottleneck TSP (BTSP) problem.
        Returns a tour as a list of city indices starting and ending at city 0.

        Minimizes the maximum edge weight along the tour using a CP-SAT model with
        a Circuit constraint and an objective on the maximum selected edge rank.
        """
        # Basic validation and small cases
        if not problem:
            return []
        n = len(problem)
        if n == 1:
            return [0, 0]
        if n == 2:
            return [0, 1, 0]

        dist = problem  # alias

        # Build unique undirected edge weights and rank them
        # Map each undirected edge (i<j) to a rank (0..K-1)
        weights: List[float] = []
        pair_to_rank: Dict[Tuple[int, int], int] = {}
        for i in range(n):
            di = dist[i]
            for j in range(i + 1, n):
                w = di[j]
                weights.append(w)

        # Sort and rank unique weights
        uniq_sorted = sorted(set(weights))
        # For speed, build a dict from weight to rank
        weight_to_rank: Dict[float, int] = {w: r for r, w in enumerate(uniq_sorted)}

        for i in range(n):
            di = dist[i]
            for j in range(i + 1, n):
                pair_to_rank[(i, j)] = weight_to_rank[di[j]]

        # Build a quick greedy tour (nearest neighbor) to get an upper bound on bottleneck
        tour = self._nearest_neighbor_tour(dist)
        ub_weight = 0.0
        for a, b in zip(tour[:-1], tour[1:]):
            w = dist[a][b]
            if w > ub_weight:
                ub_weight = w
        ub_rank = weight_to_rank[ub_weight]

        # Create CP-SAT model
        model = cp_model.CpModel()

        # Create arc variables (directed) only for edges with rank <= ub_rank
        # This prunes the search space to edges that could be in an optimal tour.
        x: Dict[Tuple[int, int], cp_model.IntVar] = {}
        arcs: List[Tuple[int, int, cp_model.IntVar]] = []
        for i in range(n):
            di = dist[i]
            for j in range(n):
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                r = pair_to_rank[(a, b)]
                if r <= ub_rank:
                    var = model.NewBoolVar(f"x_{i}_{j}")
                    x[(i, j)] = var
                    arcs.append((i, j, var))

        # Ensure that each node has at least one potential outgoing and incoming arc.
        # This guards against pathological cases where pruning removes feasibility.
        for i in range(n):
            if not any((i, j) in x for j in range(n) if j != i):
                # If this happens, fall back to the greedy tour immediately
                # (should not happen because the greedy tour edges are included)
                return tour
            if not any((j, i) in x for j in range(n) if j != i):
                return tour

        # Add Circuit constraint to force a single Hamiltonian cycle
        model.AddCircuit(arcs)

        # Integer var representing the maximum edge rank selected (bottleneck rank)
        L = model.NewIntVar(0, ub_rank, "L")

        # If arc (i->j) is selected with rank r, then enforce L >= r
        # Using reified inequalities for strong propagation.
        for (i, j), var in x.items():
            a, b = (i, j) if i < j else (j, i)
            r = pair_to_rank[(a, b)]
            if r > 0:
                model.Add(L >= r).OnlyEnforceIf(var)
            # If r == 0, constraint is redundant

        # Objective: minimize the maximum selected edge rank
        model.Minimize(L)

        # Provide a warm start (hint) from the greedy tour
        hint_vars: List[cp_model.IntVar] = []
        hint_vals: List[int] = []
        for i in range(n - 1):
            a, b = tour[i], tour[i + 1]
            if (a, b) in x:
                hint_vars.append(x[(a, b)])
                hint_vals.append(1)
        # Complete the cycle is already included in tour
        if (tour[-2], tour[-1]) in x:
            pass  # already added above as last pair of tour
        # Also hint non-used arcs to 0 (optional, small subset for efficiency)
        # We avoid setting hints for all zeros to keep hint size reasonable.

        hint_vars.append(L)
        hint_vals.append(ub_rank)
        for var, val in zip(hint_vars, hint_vals):
            model.AddHint(var, val)

        # Solve
        solver = cp_model.CpSolver()
        # Use all available cores
        workers = os.cpu_count() or 1
        solver.parameters.num_search_workers = max(1, workers)
        # A small level of linearization may help. Default is fine.

        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Fallback: return greedy (shouldn't happen for well-formed complete graphs)
            return tour

        # Extract the successor for each node
        succ = [-1] * n
        for (i, j), var in x.items():
            if solver.BooleanValue(var):
                succ[i] = j

        # Reconstruct the tour starting from node 0
        path = [0]
        visited = {0}
        current = 0
        for _ in range(n - 1):
            nxt = succ[current]
            if nxt == -1:
                # Fallback if something went wrong
                return tour
            path.append(nxt)
            if nxt in visited:
                break
            visited.add(nxt)
            current = nxt

        # Ensure it's properly closed and has all nodes once
        # If the path is short due to unexpected issues, fall back to greedy
        if len(path) != n or path[0] != 0:
            # try to complete by following successors
            seen = set(path)
            cur = path[-1]
            while len(seen) < n:
                nxt = succ[cur]
                if nxt in seen or nxt == -1:
                    break
                path.append(nxt)
                seen.add(nxt)
                cur = nxt
            if len(path) != n:
                return tour

        # Close the tour
        path.append(0)
        return path

    @staticmethod
    def _nearest_neighbor_tour(dist: List[List[float]]) -> List[int]:
        """Construct a feasible TSP tour via nearest neighbor heuristic starting at 0."""
        n = len(dist)
        unvisited = set(range(1, n))
        tour = [0]
        current = 0
        while unvisited:
            next_city = min(unvisited, key=lambda j: dist[current][j])
            unvisited.remove(next_city)
            tour.append(next_city)
            current = next_city
        tour.append(0)
        return tour