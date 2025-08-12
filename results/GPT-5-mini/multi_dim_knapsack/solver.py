from typing import Any, List, Tuple

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Multi-dimensional 0/1 knapsack solver.
        Accepts either an object with attributes (value, demand, supply) or a tuple/list
        (value, demand, supply). Returns a list of selected item indices (0-based).
        Strategy:
        - Preprocess impossible items and items with zero normalized weight.
        - Sort remaining items by value per normalized weight.
        - Compute a greedy feasible solution for an initial lower bound.
        - Branch-and-bound (iterative stack) using a fractional (relaxed) upper bound.
        """
        # Parse problem input
        try:
            if hasattr(problem, "value") and hasattr(problem, "demand") and hasattr(problem, "supply"):
                values = list(problem.value)
                demands = [list(d) for d in problem.demand]
                supply = list(problem.supply)
            else:
                values, demands, supply = problem
                values = list(values)
                demands = [list(d) for d in demands]
                supply = list(supply)
        except Exception:
            return []

        n = len(values)
        if n == 0:
            return []

        k = len(supply)
        # Validate demand dimensions
        for d in demands:
            if len(d) != k:
                return []

        # Convert supplies to floats and identify positive-supply resources
        supply_f: List[float] = [float(s) for s in supply]
        pos_res: List[int] = [j for j, s in enumerate(supply_f) if s > 0.0]

        eps = 1e-12

        always_include: List[int] = []
        items: List[Tuple[int, float, Tuple[float, ...], float]] = []
        always_value = 0.0

        # Preprocess items: filter impossible ones and identify zero-normalized-weight items
        for i in range(n):
            try:
                vi = float(values[i])
            except Exception:
                return []
            di_raw = demands[i]
            di = [float(x) for x in di_raw]

            # If any resource demand exceeds supply, item is impossible
            impossible = False
            for j in range(k):
                if di[j] > supply_f[j] + eps:
                    impossible = True
                    break
            if impossible:
                continue

            # Normalized weight across positive-supply resources
            w = 0.0
            for j in pos_res:
                # supply_f[j] > 0 for pos_res
                w += di[j] / supply_f[j]
            if w <= eps:
                # zero normalized weight: safe to include if positive value
                if vi > 0.0:
                    always_include.append(i)
                    always_value += vi
                # else ignore items with non-positive value and zero weight
            else:
                items.append((i, vi, tuple(di), w))

        # If no candidate items, return always-include items
        if not items:
            return sorted(set(always_include))

        # Sort items by value per normalized weight (descending), tie-breaker on value
        items.sort(key=lambda x: (-(x[1] / x[3]), -x[1]))

        m = len(items)
        idxs = [items[i][0] for i in range(m)]
        vals = [items[i][1] for i in range(m)]
        dems = [items[i][2] for i in range(m)]
        ws = [items[i][3] for i in range(m)]

        # Greedy initial solution to establish a lower bound
        rem0 = supply_f[:]
        greedy_selected: List[int] = []
        greedy_value = 0.0
        for t in range(m):
            d = dems[t]
            ok = True
            for j in range(k):
                if d[j] > rem0[j] + eps:
                    ok = False
                    break
            if ok:
                greedy_selected.append(idxs[t])
                greedy_value += vals[t]
                for j in range(k):
                    rem0[j] -= d[j]

        best_value = always_value + greedy_value
        best_set = list(greedy_selected)

        pos_res_local = pos_res
        supply_local = supply_f

        # Fractional upper bound based on normalized weights
        def fractional_upper_bound(start: int, rem_cap: List[float]) -> float:
            # Compute remaining normalized capacity
            C_rem = 0.0
            for j in pos_res_local:
                C_rem += rem_cap[j] / supply_local[j]
            val = 0.0
            used = 0.0
            for t in range(start, m):
                wt = ws[t]
                vt = vals[t]
                if wt <= eps:
                    val += vt
                    continue
                if used + wt <= C_rem + 1e-15:
                    val += vt
                    used += wt
                else:
                    remain = C_rem - used
                    if remain > 1e-15:
                        val += vt * (remain / wt)
                    break
            return val

        # DFS stack for branch-and-bound: (index, current_value, rem_capacity, chosen_indices)
        stack: List[Tuple[int, float, List[float], List[int]]] = [(0, 0.0, supply_f[:], [])]

        while stack:
            idx, cur_val, rem_cap, chosen = stack.pop()
            ub = cur_val + fractional_upper_bound(idx, rem_cap) + always_value
            if ub <= best_value + 1e-12:
                continue

            if idx >= m:
                total = cur_val + always_value
                if total > best_value + 1e-12:
                    best_value = total
                    best_set = chosen.copy()
                continue

            orig = idxs[idx]
            v = vals[idx]
            d = dems[idx]

            # Check feasibility of including the item
            feasible = True
            for j in range(k):
                if d[j] > rem_cap[j] + eps:
                    feasible = False
                    break

            if feasible:
                # Include branch
                new_rem = rem_cap[:]
                for j in range(k):
                    new_rem[j] -= d[j]
                new_val = cur_val + v
                ub_include = new_val + fractional_upper_bound(idx + 1, new_rem) + always_value
                if ub_include > best_value + 1e-12:
                    # Push exclude then include so include is explored next (LIFO order)
                    stack.append((idx + 1, cur_val, rem_cap, chosen))
                    stack.append((idx + 1, new_val, new_rem, chosen + [orig]))
                else:
                    # Prune include, try exclude
                    stack.append((idx + 1, cur_val, rem_cap, chosen))
            else:
                # Can't include, only exclude
                stack.append((idx + 1, cur_val, rem_cap, chosen))

        result_set = set(best_set) | set(always_include)
        return sorted(result_set)