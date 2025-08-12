from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    # Import OR-Tools CP-SAT
    from ortools.sat.python import cp_model
except Exception:  # pragma: no cover
    cp_model = None  # type: ignore

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[List[int]]:
        """
        Solve the VRP with K vehicles minimizing total distance, returning optimal routes.

        Strategy:
        - Handle special cases fast (K == 1 -> Held-Karp TSP for modest size; K == m -> single-customer routes)
        - Build a CP-SAT model (MTZ subtour elimination) with a good greedy initial hint.
        - Use multi-threaded CP-SAT parameters to accelerate search.
        """
        D: List[List[float]] = problem["D"]
        K: int = int(problem["K"])
        depot: int = int(problem["depot"])
        n: int = len(D)
        if n == 0:
            return []
        if not (0 <= depot < n):
            return []

        customers = [i for i in range(n) if i != depot]
        m = len(customers)

        if K <= 0:
            # No vehicles: only feasible if there are no customers.
            return [] if m == 0 else []

        # Special case: K == m -> each customer must have its own route (forced and optimal)
        if m > 0 and K == m:
            return [[depot, i, depot] for i in customers]

        # Special case: K == 1 -> exact TSP via Held-Karp DP for moderate size, else CP-SAT
        HK_THRESHOLD = 18
        if K == 1:
            if m <= HK_THRESHOLD:
                return [self._tsp_held_karp_route(D, depot)]
            # fall back to CP-SAT for larger instances

        # Build greedy initial solution as a CP-SAT hint to accelerate optimization.
        hint_routes = self._greedy_k_routes(D, K, depot)

        # Exact CP-SAT solver (MTZ) for general K.
        routes = self._solve_with_cpsat(D, K, depot, hint_routes)
        return routes

    # ----------------------- Helper Methods -----------------------

    def _tsp_held_karp_route(self, D: List[List[float]], depot: int) -> List[int]:
        """
        Solve TSP (single vehicle) exactly using Held-Karp DP and return a single route.
        Route starts and ends at depot.
        """
        n = len(D)
        nodes = [i for i in range(n) if i != depot]
        m = len(nodes)
        if m == 0:
            return [depot, depot]

        # DP[mask][j_idx] = min cost to start at depot, visit 'mask', end at nodes[j_idx]
        size = 1 << m
        INF = float("inf")
        dp = [[INF] * m for _ in range(size)]
        parent: List[List[Optional[int]]] = [[None] * m for _ in range(size)]

        # Initialize (singletons)
        for j_idx, j in enumerate(nodes):
            mask = 1 << j_idx
            dp[mask][j_idx] = D[depot][j]
            parent[mask][j_idx] = None

        # Iterate over masks
        for mask in range(size):
            if mask == 0:
                continue
            # For each last node j in mask
            sub = mask
            while sub:
                lsb = sub & -sub
                j_idx = (lsb.bit_length() - 1)
                sub ^= lsb
                prev_mask = mask ^ (1 << j_idx)
                if prev_mask == 0:
                    continue
                j = nodes[j_idx]
                pm = prev_mask
                while pm:
                    lsb2 = pm & -pm
                    i_idx = (lsb2.bit_length() - 1)
                    pm ^= lsb2
                    i = nodes[i_idx]
                    new_cost = dp[prev_mask][i_idx] + D[i][j]
                    if new_cost < dp[mask][j_idx]:
                        dp[mask][j_idx] = new_cost
                        parent[mask][j_idx] = i_idx

        full = (1 << m) - 1
        best_cost = INF
        last_idx_best: Optional[int] = None
        for j_idx, j in enumerate(nodes):
            total_cost = dp[full][j_idx] + D[j][depot]
            if total_cost < best_cost:
                best_cost = total_cost
                last_idx_best = j_idx

        # Reconstruct path
        assert last_idx_best is not None
        mask = full
        order_rev: List[int] = []
        j_idx = last_idx_best
        while j_idx is not None:
            order_rev.append(nodes[j_idx])
            pj = parent[mask][j_idx]
            if pj is None:
                break
            mask ^= 1 << j_idx
            j_idx = pj
        order_rev.reverse()
        route = [depot] + order_rev + [depot]
        return route

    def _greedy_k_routes(self, D: List[List[float]], K: int, depot: int) -> Optional[List[List[int]]]:
        """
        Construct a feasible set of K routes by greedy insertion.
        Returns list of K routes (each starts and ends at depot), or None if not feasible.
        """
        n = len(D)
        customers = [i for i in range(n) if i != depot]
        m = len(customers)
        if m < K or K <= 0:
            return None
        if m == 0:
            return [[depot, depot] for _ in range(K)]

        # Choose K seeds: K customers with smallest distance from depot
        customers_sorted = sorted(customers, key=lambda j: D[depot][j])
        seeds = customers_sorted[:K]
        visited = set(seeds)

        # Initialize K routes with one seed each
        routes: List[List[int]] = [[depot, s, depot] for s in seeds]

        # Insert remaining customers by minimal insertion cost
        for c in customers:
            if c in visited:
                continue
            best_delta = float("inf")
            best_r = -1
            best_pos = -1
            for r_idx, route in enumerate(routes):
                # try all insertion positions between consecutive nodes
                for pos in range(len(route) - 1):
                    a = route[pos]
                    b = route[pos + 1]
                    delta = D[a][c] + D[c][b] - D[a][b]
                    if delta < best_delta:
                        best_delta = delta
                        best_r = r_idx
                        best_pos = pos + 1
            if best_r == -1:
                return None  # should not happen
            routes[best_r].insert(best_pos, c)
            visited.add(c)

        # Ensure exactly K routes
        if len(routes) != K:
            return None
        return routes

    def _solve_with_cpsat(
        self,
        D: List[List[float]],
        K: int,
        depot: int,
        hint_routes: Optional[List[List[int]]] = None,
    ) -> List[List[int]]:
        """
        Exact CP-SAT model with MTZ subtour elimination, with optional initial hint.
        """
        if cp_model is None:
            # Fallback: if OR-Tools unavailable, return greedy solution (not guaranteed optimal)
            gr = hint_routes if hint_routes is not None else self._greedy_k_routes(D, K, depot)
            if gr is not None:
                return gr
            return []

        n = len(D)
        model = cp_model.CpModel()

        # Create binary arc variables x[i,j] for i != j
        x: Dict[Tuple[int, int], cp_model.IntVar] = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")

        # Degree constraints for non-depot nodes: exactly one in and one out
        for i in range(n):
            if i == depot:
                continue
            # in-degree == 1
            model.Add(sum(x[(j, i)] for j in range(n) if j != i) == 1)
            # out-degree == 1
            model.Add(sum(x[(i, j)] for j in range(n) if j != i) == 1)

        # Depot degree constraints
        model.Add(sum(x[(depot, j)] for j in range(n) if j != depot) == K)
        model.Add(sum(x[(i, depot)] for i in range(n) if i != depot) == K)

        # MTZ subtour elimination variables for non-depot nodes
        u: Dict[int, cp_model.IntVar] = {}
        for i in range(n):
            if i == depot:
                continue
            u[i] = model.NewIntVar(1, n - 1, f"u_{i}")
        # MTZ constraints
        for i in range(n):
            if i == depot:
                continue
            for j in range(n):
                if j == depot or i == j:
                    continue
                model.Add(u[i] + 1 <= u[j] + (n - 1) * (1 - x[(i, j)]))

        # Objective: minimize total distance
        model.Minimize(sum(D[i][j] * x[(i, j)] for i in range(n) for j in range(n) if i != j))

        # Provide a solution hint if available
        if hint_routes:
            for route in hint_routes:
                for a, b in zip(route, route[1:]):
                    if (a, b) in x:
                        model.AddHint(x[(a, b)], 1)

        solver = cp_model.CpSolver()
        # Use multi-threading for faster solve
        try:
            solver.parameters.num_search_workers = 8
        except Exception:
            pass

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            # Reconstruct routes from x
            routes: List[List[int]] = []
            nrange = range(n)
            # For each arc leaving depot, follow until back to depot
            starts = [j for j in nrange if j != depot and solver.Value(x[(depot, j)]) == 1]
            for start in starts:
                route = [depot, start]
                current = start
                while current != depot:
                    nxt = None
                    for k in nrange:
                        if k == current:
                            continue
                        if solver.Value(x[(current, k)]) == 1:
                            nxt = k
                            break
                    if nxt is None:
                        break
                    route.append(nxt)
                    current = nxt
                if route[-1] != depot:
                    route.append(depot)
                routes.append(route)
            if len(routes) != K:
                routes = self._reconstruct_routes_from_solution(solver, x, n, depot, K)
            return routes
        else:
            # Fall back to greedy routes if available to return something (should rarely happen).
            if hint_routes:
                return hint_routes
            gr = self._greedy_k_routes(D, K, depot)
            if gr is not None:
                return gr
            return []

    def _reconstruct_routes_from_solution(
        self,
        solver: "cp_model.CpSolver",
        x: Dict[Tuple[int, int], "cp_model.IntVar"],
        n: int,
        depot: int,
        K: int,
    ) -> List[List[int]]:
        nrange = range(n)
        nexts: Dict[int, int] = {}
        outs_from_depot: List[int] = []
        for i in nrange:
            for j in nrange:
                if i == j:
                    continue
                if solver.Value(x[(i, j)]) == 1:
                    if i == depot:
                        outs_from_depot.append(j)
                    else:
                        nexts[i] = j
        routes: List[List[int]] = []
        for start in outs_from_depot:
            route = [depot, start]
            cur = start
            while True:
                if cur == depot:
                    break
                nxt = nexts.get(cur, depot)
                route.append(nxt)
                cur = nxt
                if cur == depot:
                    break
            routes.append(route)
        return routes