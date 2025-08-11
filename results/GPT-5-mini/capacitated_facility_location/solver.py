from typing import Any, Dict, List, Optional

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the Capacitated Facility Location Problem.

        Input:
            problem: {
                "fixed_costs": [f_i],
                "capacities": [s_i],
                "demands": [d_j],
                "transportation_costs": [[c_ij]]
            }

        Output:
            {
                "objective_value": float,
                "facility_status": [bool],
                "assignments": [[0.0/1.0]]
            }
        """
        def infeasible(nf: int, nc: int) -> Dict[str, Any]:
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * nf,
                "assignments": [[0.0] * nc for _ in range(nf)],
            }

        # Parse inputs
        try:
            fixed_costs = problem["fixed_costs"]
            capacities = problem["capacities"]
            demands = problem["demands"]
            transportation_costs = problem["transportation_costs"]
        except Exception:
            return infeasible(0, 0)

        try:
            n_fac = int(len(fixed_costs))
            n_cust = int(len(demands))
        except Exception:
            return infeasible(0, 0)

        if n_fac == 0:
            if n_cust == 0:
                return {"objective_value": 0.0, "facility_status": [], "assignments": []}
            return infeasible(0, n_cust)
        if n_cust == 0:
            return {"objective_value": 0.0, "facility_status": [False] * n_fac, "assignments": [[] for _ in range(n_fac)]}

        # Validate transportation matrix
        if not isinstance(transportation_costs, list) or len(transportation_costs) != n_fac:
            return infeasible(n_fac, n_cust)
        for row in transportation_costs:
            if not isinstance(row, list) or len(row) != n_cust:
                return infeasible(n_fac, n_cust)

        # Convert to floats
        try:
            fixed_costs = [float(v) for v in fixed_costs]
            capacities = [float(v) for v in capacities]
            demands = [float(v) for v in demands]
            transportation_costs = [[float(v) for v in row] for row in transportation_costs]
        except Exception:
            return infeasible(n_fac, n_cust)

        eps = 1e-9
        # Quick infeasibility checks
        if any(d > max(capacities) + eps for d in demands):
            return infeasible(n_fac, n_cust)
        if sum(capacities) + eps < sum(demands):
            return infeasible(n_fac, n_cust)

        def compute_objective(open_fac: List[bool], assignments: List[List[float]]) -> float:
            tot = 0.0
            for i in range(n_fac):
                if open_fac[i]:
                    tot += fixed_costs[i]
                row = assignments[i]
                for j in range(n_cust):
                    if row[j] > 0.5:
                        tot += transportation_costs[i][j]
            return float(tot)

        def choose_scale(max_scale: int = 1000000) -> int:
            # pick power-of-10 scale that makes numbers near-integer
            candidates = [1, 10, 100, 1000, 10000, 100000, 1000000]
            vals: List[float] = []
            vals.extend(fixed_costs)
            vals.extend(capacities)
            vals.extend(demands)
            for r in transportation_costs:
                vals.extend(r)
            for s in candidates:
                if s > max_scale:
                    continue
                ok = True
                for v in vals:
                    if abs(round(v * s) - v * s) > 1e-6:
                        ok = False
                        break
                if ok:
                    return s
            return 1000

        # Try OR-Tools CP-SAT for exact/near-exact solution
        try:
            from ortools.sat.python import cp_model  # type: ignore

            scale = choose_scale()

            model = cp_model.CpModel()
            y = [model.NewBoolVar(f"y_{i}") for i in range(n_fac)]
            x = [[model.NewBoolVar(f"x_{i}_{j}") for j in range(n_cust)] for i in range(n_fac)]

            # assignment constraints
            for j in range(n_cust):
                model.Add(sum(x[i][j] for i in range(n_fac)) == 1)

            demand_scaled = [int(round(d * scale)) for d in demands]
            cap_scaled = [int(round(s * scale)) for s in capacities]
            for i in range(n_fac):
                model.Add(sum(demand_scaled[j] * x[i][j] for j in range(n_cust)) <= cap_scaled[i] * y[i])
                for j in range(n_cust):
                    model.Add(x[i][j] <= y[i])

            # objective
            terms = []
            for i in range(n_fac):
                fi = int(round(fixed_costs[i] * scale))
                if fi:
                    terms.append(fi * y[i])
                for j in range(n_cust):
                    cij = int(round(transportation_costs[i][j] * scale))
                    if cij:
                        terms.append(cij * x[i][j])
            model.Minimize(sum(terms) if terms else 0)

            solver = cp_model.CpSolver()

            # time limit selection
            time_limit_s: Optional[float] = None
            if "time_limit_s" in kwargs:
                try:
                    time_limit_s = float(kwargs["time_limit_s"])
                except Exception:
                    time_limit_s = None
            elif "time_limit_ms" in kwargs:
                try:
                    time_limit_s = float(int(kwargs["time_limit_ms"])) / 1000.0
                except Exception:
                    time_limit_s = None

            if time_limit_s is None:
                size = n_fac * n_cust
                if size <= 200:
                    time_limit_s = 2.0
                elif size <= 1000:
                    time_limit_s = 8.0
                elif size <= 5000:
                    time_limit_s = 20.0
                else:
                    time_limit_s = 60.0

            try:
                solver.parameters.max_time_in_seconds = max(1e-6, float(time_limit_s))
            except Exception:
                try:
                    solver.parameters.max_time_seconds = max(1e-6, float(time_limit_s))
                except Exception:
                    pass
            try:
                solver.parameters.num_search_workers = 8
            except Exception:
                pass

            status = solver.Solve(model)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                facility_status = [bool(solver.Value(y[i])) for i in range(n_fac)]
                assignments = [[0.0] * n_cust for _ in range(n_fac)]
                for i in range(n_fac):
                    for j in range(n_cust):
                        if solver.Value(x[i][j]) == 1:
                            assignments[i][j] = 1.0
                obj = compute_objective(facility_status, assignments)
                return {"objective_value": float(obj), "facility_status": facility_status, "assignments": assignments}
        except Exception:
            # OR-Tools not available or failed -> fallback
            pass

        # Greedy heuristic + local improvement
        def greedy_assign(order: List[int]) -> Optional[Dict[str, Any]]:
            assigned = [-1] * n_cust
            load = [0.0] * n_fac
            open_fac = [False] * n_fac
            for j in order:
                best_i = -1
                best_incr = float("inf")
                for i in range(n_fac):
                    if load[i] + demands[j] > capacities[i] + eps:
                        continue
                    incr = transportation_costs[i][j] + (0.0 if open_fac[i] else fixed_costs[i])
                    if incr < best_incr - 1e-12:
                        best_incr = incr
                        best_i = i
                if best_i == -1:
                    return None
                assigned[j] = best_i
                load[best_i] += demands[j]
                open_fac[best_i] = True
            assignments = [[0.0] * n_cust for _ in range(n_fac)]
            for j, i in enumerate(assigned):
                assignments[i][j] = 1.0
            return {"assigned": assigned, "load": load, "open": open_fac, "assignments": assignments}

        orders: List[List[int]] = [
            sorted(range(n_cust), key=lambda j: -demands[j]),
            sorted(range(n_cust), key=lambda j: demands[j]),
            sorted(range(n_cust), key=lambda j: min(transportation_costs[i][j] for i in range(n_fac))),
            list(range(n_cust)),
        ]
        try:
            cost_spread = [max(transportation_costs[i][j] for i in range(n_fac)) - min(transportation_costs[i][j] for i in range(n_fac)) for j in range(n_cust)]
            orders.append(sorted(range(n_cust), key=lambda j: -cost_spread[j]))
        except Exception:
            pass

        best_obj = float("inf")
        best_solution: Optional[Dict[str, Any]] = None

        for ordr in orders:
            sol = greedy_assign(ordr)
            if sol is None:
                continue
            assigned_local = sol["assigned"]
            load_local = sol["load"]
            open_local = sol["open"]
            assignments_local = sol["assignments"]

            improved = True
            iters = 0
            max_iters = max(200, n_cust * 5)
            while improved and iters < max_iters:
                improved = False
                iters += 1
                for j in range(n_cust):
                    cur_i = assigned_local[j]
                    cur_cost = transportation_costs[cur_i][j]
                    closing_saving = fixed_costs[cur_i] if (load_local[cur_i] - demands[j] <= eps) else 0.0
                    best_delta = 0.0
                    best_target = cur_i
                    for i in range(n_fac):
                        if i == cur_i:
                            continue
                        if load_local[i] + demands[j] > capacities[i] + eps:
                            continue
                        extra_open = 0.0 if open_local[i] else fixed_costs[i]
                        new_cost = transportation_costs[i][j] + extra_open
                        delta = new_cost - cur_cost - closing_saving
                        if delta < best_delta - 1e-9:
                            best_delta = delta
                            best_target = i
                    if best_target != cur_i:
                        prev = cur_i
                        assigned_local[j] = best_target
                        assignments_local[prev][j] = 0.0
                        assignments_local[best_target][j] = 1.0
                        load_local[prev] -= demands[j]
                        load_local[best_target] += demands[j]
                        if load_local[prev] <= eps:
                            open_local[prev] = False
                        open_local[best_target] = True
                        improved = True
                        break

            # try closing facilities if beneficial
            for i in range(n_fac):
                if not open_local[i]:
                    continue
                customers = [j for j in range(n_cust) if assigned_local[j] == i]
                if not customers:
                    open_local[i] = False
                    continue
                feasible = True
                reassign_plan = {}
                extra_cost = 0.0
                tmp_load = load_local[:]  # copy
                for j in customers:
                    best_alt = None
                    best_alt_cost = float("inf")
                    for k in range(n_fac):
                        if k == i:
                            continue
                        if tmp_load[k] + demands[j] > capacities[k] + eps:
                            continue
                        alt_cost = transportation_costs[k][j] + (0.0 if open_local[k] else fixed_costs[k])
                        if alt_cost < best_alt_cost:
                            best_alt_cost = alt_cost
                            best_alt = k
                    if best_alt is None:
                        feasible = False
                        break
                    reassign_plan[j] = (best_alt, best_alt_cost)
                    tmp_load[best_alt] += demands[j]
                    tmp_load[i] -= demands[j]
                    extra_cost += best_alt_cost
                if not feasible:
                    continue
                current_cost = sum(transportation_costs[i][j] for j in customers) + fixed_costs[i]
                new_cost = extra_cost
                if new_cost + 1e-9 < current_cost:
                    for j, (k, _) in reassign_plan.items():
                        prev = assigned_local[j]
                        assigned_local[j] = k
                        assignments_local[prev][j] = 0.0
                        assignments_local[k][j] = 1.0
                    load_local = tmp_load
                    open_local[i] = False

            total = compute_objective(open_local, assignments_local)
            if total < best_obj - 1e-12:
                best_obj = total
                best_solution = {
                    "objective_value": float(total),
                    "facility_status": [bool(v) for v in open_local],
                    "assignments": assignments_local,
                }

        if best_solution is not None:
            return best_solution

        return infeasible(n_fac, n_cust)