from __future__ import annotations

import os
from typing import Any, List, Tuple

from ortools.sat.python import cp_model


class Solver:
    def __init__(self) -> None:
        # Nothing heavy here; initialization overhead doesn't count anyway.
        pass

    @staticmethod
    def _clarke_wright_initial_routes(D: List[List[float]], depot: int, K: int) -> List[List[int]]:
        """
        Generate K routes using a generalized Clarke-Wright savings heuristic (without capacity).
        It merges singleton routes by descending savings until only K routes remain.
        Returns a list of routes where each route is [depot, nodes..., depot].
        """
        n = len(D)
        nodes = [i for i in range(n) if i != depot]
        if not nodes:
            return [[depot, depot] for _ in range(K)]

        # Initialize each node in its own route
        routes: List[List[int]] = [[i] for i in nodes]  # each is just [i] (excluding depot explicit)
        route_id_of = {i: idx for idx, i in enumerate(nodes)}

        # Precompute savings S(i, j) = d(i,0) + d(0,j) - d(i,j)
        # Only compute for i != j in nodes
        savings: List[Tuple[float, int, int]] = []
        d0 = D[depot]
        for i in nodes:
            Di = D[i]
            di0 = d0[i]
            for j in nodes:
                if i == j:
                    continue
                # Note: Distances are non-negative; off-diagonal assumed >0 by problem checks.
                s = di0 + d0[j] - Di[j]
                savings.append((s, i, j))
        # Sort descending by savings; ties arbitrary
        savings.sort(key=lambda x: x[0], reverse=True)

        # Helper functions
        def is_endpoint(route: List[int], node: int) -> bool:
            return route and (route[0] == node or route[-1] == node)

        def get_end_type(route: List[int], node: int) -> str:
            if route[0] == node:
                return "head"
            if route[-1] == node:
                return "tail"
            return "none"

        # Current number of routes
        num_routes = len(routes)

        # Process merges while we have more than K routes
        # If merges with positive savings are exhausted and still > K, continue merging with smallest penalty
        idx = 0
        max_iter = len(savings)
        while num_routes > K and idx < max_iter:
            _, i, j = savings[idx]
            idx += 1

            ri = route_id_of[i]
            rj = route_id_of[j]
            if ri == rj:
                continue  # same route; skip

            route_i = routes[ri]
            route_j = routes[rj]
            if not route_i or not route_j:
                continue

            # i and j must be endpoints of their respective routes
            if not is_endpoint(route_i, i) or not is_endpoint(route_j, j):
                continue

            # Determine orientations to connect i (tail) -> j (head)
            i_pos = get_end_type(route_i, i)
            j_pos = get_end_type(route_j, j)

            # Compute the new route arrangement based on endpoints
            # We want i to be tail of route_i and j to be head of route_j; reverse as necessary
            if i_pos == "head":
                route_i = list(reversed(route_i))
                i_pos = "tail"
            if j_pos == "tail":
                route_j = list(reversed(route_j))
                j_pos = "head"

            # Now route_i tail connects to route_j head
            # To prevent creating a cycle that doesn't include depot, we ensure no other endpoints conflict.
            new_route = route_i + route_j

            # Commit merge: replace route_i with new_route, empty route_j, update mapping
            routes[ri] = new_route
            routes[rj] = []
            for node in route_j:
                route_id_of[node] = ri
            num_routes -= 1

        # If still more than K routes (unlikely), merge arbitrarily cheapest endpoints
        if num_routes > K:
            # Build a list of endpoints
            endpoint_list: List[Tuple[int, int]] = []  # (route_id, node)
            for rid, r in enumerate(routes):
                if not r:
                    continue
                endpoint_list.append((rid, r[0]))
                if r[-1] != r[0]:  # for singleton it's the same
                    endpoint_list.append((rid, r[-1]))

            # Greedyly merge remaining routes arbitrarily by minimal added cost
            import heapq

            # Build a heap of possible merges (approx): connect tail->head for all pairs
            heap: List[Tuple[float, int, int]] = []
            alive = [bool(r) for r in routes]
            for a in range(len(routes)):
                if not alive[a] or not routes[a]:
                    continue
                for b in range(len(routes)):
                    if a == b or not alive[b] or not routes[b]:
                        continue
                    ai = routes[a][-1]  # tail of a
                    bj = routes[b][0]   # head of b
                    cost_increase = D[ai][bj] - D[ai][depot] - D[depot][bj]
                    heap.append((cost_increase, a, b))
            heapq.heapify(heap)

            while num_routes > K and heap:
                _, a, b = heapq.heappop(heap)
                if a == b:
                    continue
                if not alive[a] or not alive[b]:
                    continue
                if not routes[a] or not routes[b]:
                    continue
                # Merge a.tail -> b.head
                ai = routes[a][-1]
                bj = routes[b][0]
                # Merge
                routes[a] = routes[a] + routes[b]
                for node in routes[b]:
                    route_id_of[node] = a
                routes[b] = []
                alive[b] = False
                num_routes -= 1

        # Extract final non-empty routes and ensure exactly K by adding empty depot-depot if necessary
        final_paths = [r for r in routes if r]
        # If less than K, fill with empty routes (just depot to depot)
        while len(final_paths) < K:
            final_paths.append([])

        # Convert to full routes including depot
        full_routes: List[List[int]] = []
        for path in final_paths[:K]:
            if path:
                full_routes.append([depot] + path + [depot])
            else:
                full_routes.append([depot, depot])

        return full_routes

    def solve(self, problem: dict[str, Any], **kwargs) -> list[list[int]]:
        """
        Solve the VRP problem to optimality using CP-SAT with MTZ subtour elimination.
        Adds a strong initial solution via a Clarke-Wright style heuristic to speed up solving.

        :param problem: Dict with "D", "K", and "depot".
        :return: A list of K routes, each a list of nodes starting and ending at the depot.
        """
        D: List[List[float]] = problem["D"]
        K: int = problem["K"]
        depot: int = problem["depot"]
        n = len(D)

        # Trivial case: only depot
        if n <= 1:
            return [[depot, depot] for _ in range(K)]

        model = cp_model.CpModel()

        # Create arc variables x[i][j] for i != j
        x: List[List[cp_model.IntVar | None]] = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                x[i][j] = model.NewBoolVar(f"x_{i}_{j}")

        # In/out degree constraints for non-depot nodes: exactly 1 in and 1 out
        for i in range(n):
            if i == depot:
                continue
            # incoming to i
            in_vars = [x[j][i] for j in range(n) if j != i]
            model.Add(sum(v for v in in_vars if v is not None) == 1)
            # outgoing from i
            out_vars = [x[i][j] for j in range(n) if j != i]
            model.Add(sum(v for v in out_vars if v is not None) == 1)

        # Depot degree equals K
        out_depot_vars = [x[depot][j] for j in range(n) if j != depot]
        in_depot_vars = [x[i][depot] for i in range(n) if i != depot]
        model.Add(sum(v for v in out_depot_vars if v is not None) == K)
        model.Add(sum(v for v in in_depot_vars if v is not None) == K)

        # MTZ subtour elimination: u[i] in [1, n-1] for i != depot
        u = [None] * n
        for i in range(n):
            if i == depot:
                u[i] = None
            else:
                u[i] = model.NewIntVar(1, n - 1, f"u_{i}")

        # MTZ constraints for all i != depot and j != depot, i != j:
        # u[i] + 1 <= u[j] + (n-1)*(1 - x[i][j])
        Nminus1 = n - 1
        for i in range(n):
            if i == depot:
                continue
            for j in range(n):
                if j == depot or i == j:
                    continue
                vij = x[i][j]
                if vij is not None:
                    model.Add(u[i] + 1 <= u[j] + Nminus1 * (1 - vij))

        # Objective: minimize total distance
        # CP-SAT can accept float coefficients; it will scale as needed internally.
        terms = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                vij = x[i][j]
                if vij is not None:
                    cost = D[i][j]
                    # Skip zero-cost edges if any (though problem expects >0 off-diagonals)
                    if cost != 0:
                        terms.append(cost * vij)
                    else:
                        # Still add to keep structure consistent
                        terms.append(cost * vij)
        if terms:
            model.Minimize(sum(terms))
        else:
            # Fallback: no arcs? Should not happen. Minimize 0.
            model.Minimize(0)

        # Provide an initial feasible solution via Clarke-Wright-like heuristic
        # This often dramatically speeds up optimality proof.
        try:
            init_routes = self._clarke_wright_initial_routes(D, depot, K)
            # Build hints for x and u
            hints_vars: List[cp_model.IntVar] = []
            hints_vals: List[int] = []

            # Sequence number for u variables across routes
            seq = 1
            for route in init_routes:
                # route is [depot, ..., depot]
                for idx in range(len(route) - 1):
                    a, b = route[idx], route[idx + 1]
                    if a != b:
                        v = x[a][b]
                        if v is not None:
                            hints_vars.append(v)
                            hints_vals.append(1)
                # Add u hints increasing along the route excluding depot nodes
                inner_nodes = [node for node in route if node != depot]
                for k, node in enumerate(inner_nodes):
                    if u[node] is not None:
                        hints_vars.append(u[node])
                        hints_vals.append(min(seq, n - 1))
                        seq += 1

            # Optional: set 0 for some obvious non-used arcs from depot to non-start nodes and from non-end to depot
            # But to keep hints lean, we avoid setting too many zeros.

            if hints_vars:
                model.AddHint(hints_vars, hints_vals)
        except Exception:
            # If heuristic fails for any reason, skip hinting.
            pass

        solver = cp_model.CpSolver()
        # Multi-threading to speed up
        workers = os.cpu_count() or 1
        # Avoid using too many threads on tiny instances; cap at 8
        solver.parameters.num_search_workers = max(1, min(workers, 8))
        # Keep deterministic enough without setting random seeds; no time limit to ensure optimality.

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            routes: list[list[int]] = []
            # Find all starting arcs from depot
            starts = [j for j in range(n) if j != depot and solver.Value(x[depot][j]) == 1]
            # It should be exactly K, but we iterate over found ones
            for j in starts:
                route = [depot, j]
                current = j
                visited_in_route = set([depot, j])
                # Follow until return to depot
                while current != depot:
                    found_next = False
                    for k in range(n):
                        if current == k:
                            continue
                        vij = x[current][k]
                        if vij is not None and solver.Value(vij) == 1:
                            route.append(k)
                            current = k
                            if current in visited_in_route:
                                # Safety break in case of unexpected loop
                                break
                            visited_in_route.add(current)
                            found_next = True
                            break
                    if not found_next:
                        # Should not happen; break to avoid infinite loop
                        break
                routes.append(route)

            # Ensure exactly K routes; if fewer (shouldn't happen), pad with depot-only
            while len(routes) < K:
                routes.append([depot, depot])

            return routes
        else:
            # Infeasible or not solved to optimality (should not happen in given tasks).
            return []