from __future__ import annotations

from array import array
from typing import Any

from ortools.sat.python import cp_model

INF = 1.0e100
BASE_CODE = -2
NONE_CODE = -1
DP_LIMIT = 9

class Solver:
    def __init__(self) -> None:
        pass

    def _solve_dp(self, D: list[list[float]], K: int, depot: int) -> list[list[int]]:
        n = len(D)
        customers = [i for i in range(n) if i != depot]
        m = len(customers)

        if m == 0:
            return [] if K == 0 else []
        if K <= 0 or K > m:
            return []

        dist0 = [float(D[depot][node]) for node in customers]
        dist_cc = [[float(D[a][b]) for b in customers] for a in customers]

        size = 1 << m
        full_mask = size - 1
        total = size * m

        popcount = [0] * size
        for mask in range(1, size):
            popcount[mask] = popcount[mask >> 1] + (mask & 1)

        bit_to_idx = [-1] * size
        for i in range(m):
            bit_to_idx[1 << i] = i

        parents: list[array[int] | None] = [None] * (K + 1)
        dp_prev = [INF] * total

        for k in range(1, K + 1):
            dp_cur = [INF] * total
            par = array("h", [NONE_CODE]) * total
            parents[k] = par

            for mask in range(1, size):
                if popcount[mask] < k:
                    continue

                base = mask * m
                bits_j = mask
                while bits_j:
                    lsb_j = bits_j & -bits_j
                    j = bit_to_idx[lsb_j]
                    prev_mask = mask ^ lsb_j

                    best = INF
                    code = NONE_CODE

                    if prev_mask == 0:
                        if k == 1:
                            best = dist0[j]
                            code = BASE_CODE
                    else:
                        prev_base = prev_mask * m
                        bits_i = prev_mask
                        while bits_i:
                            lsb_i = bits_i & -bits_i
                            i = bit_to_idx[lsb_i]
                            idx = prev_base + i

                            val = dp_cur[idx] + dist_cc[i][j]
                            if val < best:
                                best = val
                                code = i

                            if k > 1:
                                val = dp_prev[idx] + dist0[i] + dist0[j]
                                if val < best:
                                    best = val
                                    code = i + m

                            bits_i ^= lsb_i

                    idx = base + j
                    dp_cur[idx] = best
                    par[idx] = code
                    bits_j ^= lsb_j

            dp_prev = dp_cur

        best_j = -1
        best_cost = INF
        base = full_mask * m
        for j in range(m):
            val = dp_prev[base + j] + dist0[j]
            if val < best_cost:
                best_cost = val
                best_j = j

        if best_j < 0 or best_cost >= INF / 2:
            return []

        rev_nodes: list[int] = []
        rev_starts: list[bool] = []
        mask = full_mask
        k = K
        j = best_j

        while True:
            code = parents[k][mask * m + j]
            rev_nodes.append(customers[j])
            rev_starts.append(code == BASE_CODE or code >= m)

            if code == BASE_CODE:
                break

            mask ^= 1 << j
            if code < m:
                j = code
            else:
                j = code - m
                k -= 1

        routes: list[list[int]] = []
        current: list[int] = []
        for node, start in zip(rev_nodes[::-1], rev_starts[::-1]):
            if start:
                if current:
                    current.append(depot)
                    routes.append(current)
                current = [depot, node]
            else:
                current.append(node)

        if current:
            current.append(depot)
            routes.append(current)

        return routes if len(routes) == K else []

    def _solve_cpsat(self, D: list[list[float]], K: int, depot: int) -> list[list[int]]:
        n = len(D)
        order = [depot] + [i for i in range(n) if i != depot]

        model = cp_model.CpModel()
        x: list[list[cp_model.IntVar | None]] = [[None] * n for _ in range(n)]
        arcs: list[tuple[int, int, cp_model.IntVar]] = []
        obj_terms = []
        depot_out = []

        for i in range(n):
            row = x[i]
            drow = D[order[i]]
            for j in range(n):
                if i == j:
                    continue
                lit = model.NewBoolVar(f"x_{i}_{j}")
                row[j] = lit
                arcs.append((i, j, lit))
                obj_terms.append(drow[order[j]] * lit)
                if i == 0:
                    depot_out.append(lit)

        model.AddMultipleCircuit(arcs)
        model.Add(sum(depot_out) == K)
        model.Minimize(sum(obj_terms))

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            return []

        succ = [-1] * n
        starts: list[int] = []
        for i in range(n):
            row = x[i]
            if i == 0:
                for j in range(1, n):
                    lit = row[j]
                    if lit is not None and solver.Value(lit):
                        starts.append(j)
            else:
                for j in range(n):
                    if i != j:
                        lit = row[j]
                        if lit is not None and solver.Value(lit):
                            succ[i] = j
                            break

        routes: list[list[int]] = []
        for start in starts:
            route = [depot]
            cur = start
            while cur != 0:
                route.append(order[cur])
                cur = succ[cur]
                if cur < 0:
                    return []
            route.append(depot)
            routes.append(route)

        return routes

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> list[list[int]]:
        D = problem["D"]
        K = int(problem["K"])
        depot = int(problem["depot"])

        n = len(D)
        m = n - 1

        if n == 0:
            return []
        if K < 0:
            return []
        if n == 1:
            return [] if K == 0 else []
        if K == 0 or K > m:
            return []

        if m <= DP_LIMIT:
            ans = self._solve_dp(D, K, depot)
            if ans:
                return ans

        return self._solve_cpsat(D, K, depot)