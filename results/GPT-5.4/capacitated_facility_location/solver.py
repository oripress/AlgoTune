from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

class Solver:
    def __init__(self) -> None:
        self._eps = 1e-12

    @staticmethod
    def _infeasible_solution(n: int, m: int) -> dict[str, Any]:
        return {
            "objective_value": float("inf"),
            "facility_status": [False] * n,
            "assignments": [[0.0] * m for _ in range(n)],
        }

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        fixed_costs = np.asarray(problem["fixed_costs"], dtype=np.float64)
        capacities = np.asarray(problem["capacities"], dtype=np.float64)
        demands = np.asarray(problem["demands"], dtype=np.float64)
        transportation_costs = np.asarray(
            problem["transportation_costs"], dtype=np.float64
        )

        n = int(fixed_costs.size)
        m = int(demands.size)

        if n == 0:
            return self._infeasible_solution(n, m) if m else {
                "objective_value": 0.0,
                "facility_status": [],
                "assignments": [],
            }

        if transportation_costs.shape != (n, m):
            transportation_costs = transportation_costs.reshape(n, m)

        if m == 0:
            return {
                "objective_value": 0.0,
                "facility_status": [False] * n,
                "assignments": [[] for _ in range(n)],
            }

        eps = self._eps
        total_demand = float(demands.sum())
        cap_sum = float(capacities.sum())
        if total_demand > cap_sum + 1e-9:
            return self._infeasible_solution(n, m)

        max_capacity = float(capacities.max(initial=0.0))
        if np.any(demands > max_capacity + 1e-9):
            return self._infeasible_solution(n, m)

        assignments = np.zeros((n, m), dtype=np.float64)
        active = np.ones(m, dtype=bool)
        residual_cap = capacities.copy()
        residual_cap = capacities.copy()

        # Propagate customers that currently have only one feasible facility.
        while True:
            active_idx = np.flatnonzero(active)
            if active_idx.size == 0:
                break

            feasible_now = residual_cap[:, None] + 1e-9 >= demands[active_idx][None, :]
            counts = feasible_now.sum(axis=0)
            if np.any(counts == 0):
                return self._infeasible_solution(n, m)

            unique_pos = np.flatnonzero(counts == 1)
            if unique_pos.size == 0:
                break

            unique_customers = active_idx[unique_pos]
            unique_facilities = feasible_now[:, unique_pos].argmax(axis=0).astype(
                np.int32, copy=False
            )

            load_add = np.zeros(n, dtype=np.float64)
            np.add.at(load_add, unique_facilities, demands[unique_customers])
            if np.any(load_add > residual_cap + 1e-9):
                return self._infeasible_solution(n, m)

            residual_cap -= load_add
            assignments[unique_facilities, unique_customers] = 1.0
            assigned_fac[unique_customers] = unique_facilities
            active[unique_customers] = False

        forced_open = assignments.any(axis=1)

        active_idx = np.flatnonzero(active)
        if active_idx.size == 0:
            objective_value = float(
                fixed_costs @ forced_open.astype(np.float64)
                + np.sum(transportation_costs * assignments)
            )
            return {
                "objective_value": objective_value,
                "facility_status": forced_open.tolist(),
                "assignments": assignments.tolist(),
            }

        active_demands = demands[active_idx]
        feasible = residual_cap[:, None] + 1e-9 >= active_demands[None, :]
        if np.any(feasible.sum(axis=0) == 0):
            return self._infeasible_solution(n, m)

        used_facilities = forced_open | feasible.any(axis=1)
        facility_idx = np.flatnonzero(used_facilities)
        active_fac_count = int(facility_idx.size)

        fixed_costs_sub = fixed_costs[facility_idx]
        residual_sub = residual_cap[facility_idx]
        forced_open_sub = forced_open[facility_idx]
        feasible_sub = feasible[facility_idx]

        fac_local, cust_local = np.nonzero(feasible_sub)
        pair_count = int(fac_local.size)

        cust_count = int(active_idx.size)
        num_vars = active_fac_count + pair_count

        obj = np.empty(num_vars, dtype=np.float64)
        obj[:active_fac_count] = fixed_costs_sub
        obj[active_fac_count:] = transportation_costs[
            facility_idx[fac_local], active_idx[cust_local]
        ]

        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []

        # Each remaining customer assigned exactly once.
        row_parts.append(cust_local.astype(np.int32, copy=False))
        col_parts.append(active_fac_count + np.arange(pair_count, dtype=np.int32))
        data_parts.append(np.ones(pair_count, dtype=np.float64))

        # Capacity constraints on residual capacities.
        cap_rows = cust_count + np.arange(active_fac_count, dtype=np.int32)
        row_parts.append(cap_rows)
        col_parts.append(np.arange(active_fac_count, dtype=np.int32))
        data_parts.append(-residual_sub)

        row_parts.append(cap_rows[fac_local])
        col_parts.append(active_fac_count + np.arange(pair_count, dtype=np.int32))
        data_parts.append(active_demands[cust_local])

        # Global demand cover inequality.
        cover_row = cust_count + active_fac_count
        row_parts.append(np.full(active_fac_count, cover_row, dtype=np.int32))
        col_parts.append(np.arange(active_fac_count, dtype=np.int32))
        data_parts.append(residual_sub)

        num_rows = cover_row + 1

        # Link only zero-demand assignments to non-forced-open facilities.
        zero_demand_local = active_demands <= eps
        link_mask = zero_demand_local[cust_local] & (~forced_open_sub[fac_local])
        if np.any(link_mask):
            link_count = int(link_mask.sum())
            link_rows = num_rows + np.arange(link_count, dtype=np.int32)
            link_fac = fac_local[link_mask].astype(np.int32, copy=False)
            link_pair = np.flatnonzero(link_mask).astype(np.int32, copy=False)

            row_parts.append(link_rows)
            col_parts.append(link_fac)
            data_parts.append(-np.ones(link_count, dtype=np.float64))

            row_parts.append(link_rows)
            col_parts.append(active_fac_count + link_pair)
            data_parts.append(np.ones(link_count, dtype=np.float64))

            num_rows += link_count

        rows = np.concatenate(row_parts)
        cols = np.concatenate(col_parts)
        data = np.concatenate(data_parts)

        A = coo_matrix((data, (rows, cols)), shape=(num_rows, num_vars)).tocsr()

        lb = np.full(num_rows, -np.inf, dtype=np.float64)
        ub = np.zeros(num_rows, dtype=np.float64)
        lb[:cust_count] = 1.0
        ub[:cust_count] = 1.0
        lb[cover_row] = float(active_demands.sum())
        ub[cover_row] = np.inf

        var_lb = np.zeros(num_vars, dtype=np.float64)
        var_ub = np.ones(num_vars, dtype=np.float64)
        var_lb[:active_fac_count] = forced_open_sub.astype(np.float64)

        bounds = Bounds(lb=var_lb, ub=var_ub)
        integrality = np.ones(num_vars, dtype=np.int8)

        try:
            res = milp(
                c=obj,
                integrality=integrality,
                bounds=bounds,
                constraints=LinearConstraint(A, lb, ub),
                options={"disp": False, "mip_rel_gap": 0.009},
            )
        except Exception:
            return self._infeasible_solution(n, m)

        if not getattr(res, "success", False) or res.x is None:
            return self._infeasible_solution(n, m)

        sol = np.clip(np.round(res.x), 0.0, 1.0)
        y_sub = sol[:active_fac_count].astype(bool)
        pair_vals = sol[active_fac_count:]

        assignments[facility_idx[fac_local], active_idx[cust_local]] = pair_vals

        status = np.zeros(n, dtype=bool)
        status[facility_idx] = y_sub
        status |= forced_open

        # Conservative repair if rounding leaves inconsistency.
        loads = assignments @ demands
        if (assignments.sum(axis=0) != 1).any() or (loads > capacities + 1e-9).any():
            assignments.fill(0.0)
            status[:] = False
            remaining = capacities.copy()
            order = np.argsort(-demands)
            for j in order:
                feasible_j = np.flatnonzero(remaining + 1e-9 >= demands[j])
                if feasible_j.size == 0:
                    return self._infeasible_solution(n, m)
                scores = transportation_costs[feasible_j, j] + (
                    ~status[feasible_j]
                ) * fixed_costs[feasible_j]
                i = int(feasible_j[int(np.argmin(scores))])
                assignments[i, j] = 1.0
                status[i] = True
                remaining[i] -= demands[j]

        objective_value = float(
            fixed_costs @ status.astype(np.float64)
            + np.sum(transportation_costs * assignments)
        )

        return {
            "objective_value": objective_value,
            "facility_status": status.tolist(),
            "assignments": assignments.tolist(),
        }