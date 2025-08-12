from __future__ import annotations

from typing import Any, List, Sequence, Tuple

try:
    # OR-Tools is available per task description
    from ortools.sat.python import cp_model
except Exception:  # pragma: no cover
    cp_model = None  # Fallback handled during solve

class Solver:
    def _parse_problem(
        self, problem: Any
    ) -> Tuple[List[int], List[List[int]], List[int]]:
        """
        Accepts:
          - (value, demand, supply)
          - an object with attributes: value, demand, supply

        Returns:
          (value, demand, supply) as lists of ints
        """
        # If it's a tuple/list length 3
        if isinstance(problem, (tuple, list)) and len(problem) == 3:
            value, demand, supply = problem
        else:
            # Try attribute access
            try:
                value = getattr(problem, "value")
                demand = getattr(problem, "demand")
                supply = getattr(problem, "supply")
            except Exception as exc:
                raise ValueError(
                    "Problem must be (value, demand, supply) or have those attributes."
                ) from exc

        # Convert to lists
        value = list(value)
        demand = [list(row) for row in demand]
        supply = list(supply)

        # Ensure integral types (ORTools CP-SAT expects integers)
        def _safe_int(x):
            # Only cast floats that are very close to integers
            if isinstance(x, bool):
                return int(x)
            if isinstance(x, int):
                return x
            if isinstance(x, float):
                xi = int(round(x))
                if abs(x - xi) < 1e-9:
                    return xi
                raise ValueError("Non-integer values not supported by this solver.")
            try:
                return int(x)
            except Exception as exc:
                raise ValueError("Values must be integers.") from exc

        value = [_safe_int(v) for v in value]
        supply = [_safe_int(s) for s in supply]
        k = len(supply)

        # Normalize demand row lengths to k (pad/truncate)
        norm_demand: List[List[int]] = []
        for row in demand:
            if len(row) < k:
                row = list(row) + [0] * (k - len(row))
            elif len(row) > k:
                row = list(row[:k])
            norm_demand.append([_safe_int(x) for x in row])

        return value, norm_demand, supply

    @staticmethod
    def _greedy_by_weights(
        values: Sequence[int],
        demands: Sequence[Sequence[int]],
        supply: Sequence[int],
        weights: Sequence[float],
    ) -> Tuple[int, List[int]]:
        k = len(supply)
        n = len(values)
        # Compute density scores
        dens = []
        for i in range(n):
            di = demands[i]
            denom = 0.0
            for r in range(k):
                denom += di[r] * weights[r]
            if denom <= 0.0:
                score = float("inf") if values[i] > 0 else -float("inf")
            else:
                score = float(values[i]) / denom
            dens.append((score, i))
        dens.sort(key=lambda t: t[0], reverse=True)
        order = [i for _, i in dens]
        return Solver._greedy_lb(values, demands, supply, order)

    @staticmethod
    def _greedy_lb(
        values: Sequence[int],
        demands: Sequence[Sequence[int]],
        supply: Sequence[int],
        order: Sequence[int] | None = None,
    ) -> Tuple[int, List[int]]:
        """
        Simple greedy by value / weighted demand ratio as a fast feasible lower bound.
        Returns (value, selected_indices_relative).
        """
        n = len(values)
        k = len(supply)
        if n == 0:
            return 0, []
        if order is None:
            # Compute weights inversely proportional to supply
            weights = [1.0 / (s if s > 0 else 1.0) for s in supply]
            dens = []
            for i in range(n):
                di = demands[i]
                denom = sum(di[r] * weights[r] for r in range(k))
                if denom <= 0.0:
                    score = float("inf") if values[i] > 0 else -float("inf")
                else:
                    score = float(values[i]) / denom
                dens.append((score, i))
            dens.sort(key=lambda t: t[0], reverse=True)
            order = [i for _, i in dens]
        sel: List[int] = []
        used = [0] * k
        total = 0
        for i in order:
            di = demands[i]
            feasible = True
            for r in range(k):
                if used[r] + di[r] > supply[r]:
                    feasible = False
                    break
            if feasible and values[i] > 0:
                sel.append(i)
                total += values[i]
                for r in range(k):
                    used[r] += di[r]
        return total, sel

    @staticmethod
    def _improve_local(
        values: Sequence[int],
        demands: Sequence[Sequence[int]],
        supply: Sequence[int],
        initial_sel: List[int],
        max_unselected_considered: int = 200,
    ) -> Tuple[int, List[int]]:
        """
        Try simple local improvements:
        - add feasible unselected items
        - 1-for-1 swap: replace one selected with one unselected if value improves and feasibility holds
        Returns improved (value, selection).
        """
        n = len(values)
        k = len(supply)
        sel_set = set(initial_sel)
        used = [0] * k
        total = 0
        for i in initial_sel:
            total += values[i]
            di = demands[i]
            for r in range(k):
                used[r] += di[r]

        # Build an order of unselected candidates by density (using inverse supply)
        inv_sup = [1.0 / (s if s > 0 else 1.0) for s in supply]
        dens = []
        for i in range(n):
            if i in sel_set:
                continue
            di = demands[i]
            denom = 0.0
            for r in range(k):
                denom += di[r] * inv_sup[r]
            if denom <= 0.0:
                score = float("inf") if values[i] > 0 else -float("inf")
            else:
                score = float(values[i]) / denom
            dens.append((score, i))
        dens.sort(key=lambda t: t[0], reverse=True)
        unselected_order = [i for _, i in dens[:max_unselected_considered]]

        improved = True
        sel_list = list(initial_sel)
        # Helper to check feasibility of adding an item
        def can_add(idx: int) -> bool:
            di = demands[idx]
            for r in range(k):
                if used[r] + di[r] > supply[r]:
                    return False
            return True

        # Helper to check feasibility of replacing old -> new
        def can_swap(old_idx: int, new_idx: int) -> bool:
            do = demands[old_idx]
            dn = demands[new_idx]
            for r in range(k):
                if used[r] - do[r] + dn[r] > supply[r]:
                    return False
            return True

        # Repeat until no further improvement
        while improved:
            improved = False
            # Try to add items greedily
            added_any = False
            for j in unselected_order:
                if j in sel_set:
                    continue
                if values[j] <= 0:
                    continue
                if can_add(j):
                    # add j
                    sel_set.add(j)
                    sel_list.append(j)
                    total += values[j]
                    dj = demands[j]
                    for r in range(k):
                        used[r] += dj[r]
                    added_any = True
            if added_any:
                improved = True
                continue  # try to add further in next loop

            # Try 1-for-1 swaps
            # Sort selected by increasing value so we try replacing low-value items first
            sel_list.sort(key=lambda idx: values[idx])
            for old in sel_list:
                v_old = values[old]
                # Look for a better unselected to replace it
                for new in unselected_order:
                    if new in sel_set:
                        continue
                    if values[new] <= v_old:
                        continue
                    if can_swap(old, new):
                        # perform swap
                        sel_set.remove(old)
                        sel_set.add(new)
                        # update used
                        do = demands[old]
                        dn = demands[new]
                        for r in range(k):
                            used[r] -= do[r]
                            used[r] += dn[r]
                        total += values[new] - v_old
                        # update list
                        sel_list.remove(old)
                        sel_list.append(new)
                        improved = True
                        break
                if improved:
                    break  # restart add phase
        sel_list.sort()
        return total, sel_list

    def solve(self, problem, **kwargs) -> Any:
        """
        Returns list of selected item indices (0..n-1). Empty list on failure.

        Must be optimal to pass validation.
        """
        # Parse problem
        try:
            value, demand, supply = self._parse_problem(problem)
        except Exception:
            return []

        if cp_model is None:
            # OR-Tools not available; cannot guarantee optimal solution
            return []

        n = len(value)
        k = len(supply)

        if n == 0 or k == 0:
            return []

        # Preprocess items:
        # - Eliminate items that are infeasible individually (demand exceeds supply in any resource)
        # - Exclude items with non-positive value (they won't be chosen in a maximizing objective)
        # - Collect items with zero demand and positive value to include for free
        keep_indices: List[int] = []
        zero_demand_positive: List[int] = []
        for i in range(n):
            di = demand[i]
            # If any resource demand exceeds supply, the item can never be selected
            infeasible = False
            for r in range(k):
                if di[r] > supply[r]:
                    infeasible = True
                    break
            if infeasible:
                continue
            zero_dem = True
            for r in range(k):
                if di[r] != 0:
                    zero_dem = False
                    break
            if zero_dem:
                if value[i] > 0:
                    zero_demand_positive.append(i)
                continue
            if value[i] > 0:
                keep_indices.append(i)

        # If after preprocessing nothing remains to decide, return just the freebies
        if not keep_indices:
            return sorted(zero_demand_positive)

        # Build reduced arrays
        red_value = [value[i] for i in keep_indices]
        red_demand = [demand[i] for i in keep_indices]

        # Identify non-binding resources and drop them (sum of all demands <= supply)
        active_resources: List[int] = []
        for r in range(k):
            s = 0
            for j in range(len(red_value)):
                s += red_demand[j][r]
            if s > supply[r]:
                active_resources.append(r)
        if len(active_resources) < k:
            # Reduce demands and supply to only active resources
            new_supply = [supply[r] for r in active_resources]
            new_red_demand = []
            for j in range(len(red_value)):
                new_red_demand.append([red_demand[j][r] for r in active_resources])
            supply = new_supply
            red_demand = new_red_demand
            k = len(active_resources)

        # Quick feasibility check: if sum of demands across items <= supply per active resource,
        # then take them all.
        take_all = True
        for r in range(k):
            s = 0
            for j in range(len(red_value)):
                s += red_demand[j][r]
            if s > supply[r]:
                take_all = False
                break
        if take_all:
            selected = sorted(zero_demand_positive + keep_indices)
            return selected

        # Build multiple greedy lower bounds on reduced instance
        # Use several weight schemes to construct diverse solutions
        greedy_candidates: List[Tuple[int, List[int]]] = []

        # 1) Inverse supply
        if k > 0:
            w1 = [1.0 / (s if s > 0 else 1.0) for s in supply]
            greedy_candidates.append(self._greedy_by_weights(red_value, red_demand, supply, w1))

        # 2) Uniform weights
        w2 = [1.0] * k
        greedy_candidates.append(self._greedy_by_weights(red_value, red_demand, supply, w2))

        # 3) Scarcity squared
        w3 = [1.0 / ((s if s > 0 else 1.0) ** 2) for s in supply]
        greedy_candidates.append(self._greedy_by_weights(red_value, red_demand, supply, w3))

        # 4) Sum demand ratio (handled by uniform)
        # 5) Original _greedy_lb default (inverse supply) to ensure inclusion
        greedy_candidates.append(self._greedy_lb(red_value, red_demand, supply))

        # Choose best greedy and try local improvement
        best_val, best_sel = max(greedy_candidates, key=lambda t: t[0])
        if best_sel:
            imp_val, imp_sel = self._improve_local(red_value, red_demand, supply, best_sel)
            if imp_val > best_val:
                best_val, best_sel = imp_val, imp_sel

        greedy_val_red, greedy_sel_red = best_val, best_sel

        # Build CP-SAT model on reduced items
        model = cp_model.CpModel()
        m = len(red_value)
        x = [model.NewBoolVar(f"x_{j}") for j in range(m)]

        # Capacity constraints (only active resources)
        # Precompute per-resource coefficient vectors to speed model build
        for r in range(k):
            coeffs = [red_demand[j][r] for j in range(m)]
            model.Add(sum(x[j] * coeffs[j] for j in range(m)) <= supply[r])

        # Objective
        model.Maximize(sum(x[j] * red_value[j] for j in range(m)))

        # Add a lower bound on the objective to prune search if we have a greedy solution
        if greedy_val_red > 0:
            model.Add(sum(x[j] * red_value[j] for j in range(m)) >= int(greedy_val_red))

        # Provide solver parameters for speed
        solver = cp_model.CpSolver()
        try:
            import psutil  # type: ignore

            num_workers = psutil.cpu_count(logical=True) or 1
        except Exception:
            num_workers = 1
        solver.parameters.num_search_workers = max(1, min(8, int(num_workers)))
        # Keep presolve/probing at defaults; they are usually good.

        # Solve
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Fallback: return greedy + freebies (should at least be feasible)
            result = sorted(zero_demand_positive + [keep_indices[j] for j in greedy_sel_red])
            return result

        # Extract solution
        chosen: List[int] = []
        for j in range(m):
            if solver.Value(x[j]) == 1:
                chosen.append(keep_indices[j])

        # Merge with freebies (zero-demand positive)
        result = sorted(zero_demand_positive + chosen)
        return result