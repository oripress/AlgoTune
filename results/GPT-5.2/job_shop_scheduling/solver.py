from __future__ import annotations

import os
from typing import Any

from ortools.sat.python import cp_model

def _greedy_upper_bound(M: int, jobs_data: list[list[tuple[int, int]]]) -> int:
    """Cheap feasible makespan upper bound via list scheduling (O(J * n_ops))."""
    J = len(jobs_data)
    next_k = [0] * J
    job_ready = [0] * J
    machine_ready = [0] * M
    remaining = 0
    for job in jobs_data:
        remaining += len(job)

    while remaining:
        best_j = -1
        best_s = 1 << 60
        best_e = 1 << 60
        best_p = 1 << 60

        for j, job in enumerate(jobs_data):
            k = next_k[j]
            if k >= len(job):
                continue
            m, p = job[k]
            s = job_ready[j]
            mr = machine_ready[m]
            if mr > s:
                s = mr
            e = s + p
            if (s < best_s) or (s == best_s and (e < best_e or (e == best_e and p < best_p))):
                best_s, best_e, best_p = s, e, p
                best_j = j

        j = best_j
        k = next_k[j]
        m, p = jobs_data[j][k]
        e = best_s + p
        job_ready[j] = e
        machine_ready[m] = e
        next_k[j] = k + 1
        remaining -= 1

    ms = 0
    for t in job_ready:
        if t > ms:
            ms = t
    return ms

class Solver:
    def __init__(self) -> None:
        self._cpu = os.cpu_count() or 1

    def solve(self, problem: dict[str, Any], **kwargs) -> list[list[int]]:
        M = int(problem["num_machines"])
        jobs_data: list[list[tuple[int, int]]] = problem["jobs"]
        J = len(jobs_data)

        # Trivial stats + lower bound.
        n_ops = 0
        horizon_sum = 0
        max_job_sum = 0
        mach_load = [0] * M
        job_sum = [0] * J
        for j, job in enumerate(jobs_data):
            js = 0
            for m, p in job:
                n_ops += 1
                horizon_sum += p
                js += p
                mach_load[m] += p
            job_sum[j] = js
            if js > max_job_sum:
                max_job_sum = js

        max_mach_load = 0
        for s in mach_load:
            if s > max_mach_load:
                max_mach_load = s
        lb = max_job_sum if max_job_sum > max_mach_load else max_mach_load

        if n_ops == 0:
            return [[] for _ in range(J)]

        # Upper bound for variable domains.
        # Only compute greedy UB for large instances (amortize Python cost).
        if n_ops >= 220:
            ub = _greedy_upper_bound(M, jobs_data)
            horizon = ub if lb <= ub < horizon_sum else horizon_sum
        else:
            horizon = horizon_sum

        model = cp_model.CpModel()

        NewIntVar = model.NewIntVar
        NewFixedInterval = model.NewFixedSizeIntervalVar
        Add = model.Add
        AddNoOverlap = model.AddNoOverlap
        empty = ""

        machine_to_intervals: list[list[cp_model.IntervalVar]] = [[] for _ in range(M)]
        start_vars: list[list[cp_model.IntVar]] = [[] for _ in range(J)]
        last_ends: list[Any] = []

        for j in range(J):
            job = jobs_data[j]
            nj = len(job)
            sj = start_vars[j]

            if nj == 0:
                last_ends.append(0)
                continue
                continue

            total = job_sum[j]
            offset = horizon - total  # >= 0 because horizon >= max_job_sum >= total
            prev_end = None
            prefix = 0
            for m, p in job:
                s = NewIntVar(prefix, offset + prefix, empty)
                sj.append(s)

                machine_to_intervals[m].append(NewFixedInterval(s, p, empty))

                if prev_end is not None:
                    Add(s >= prev_end)
                prev_end = s + p
                prefix += p

            last_ends.append(prev_end)
        for m in range(M):
            itvs = machine_to_intervals[m]
            if len(itvs) > 1:
                AddNoOverlap(itvs)

        makespan = NewIntVar(lb, horizon, empty)
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        solver.parameters.random_seed = 0

        # Reduce overhead for small/medium instances.
        if n_ops < 260:
            solver.parameters.num_search_workers = 1
            solver.parameters.cp_model_presolve = False
            solver.parameters.cp_model_probing_level = 0
            solver.parameters.symmetry_level = 0
            solver.parameters.use_sat_inprocessing = False
        else:
            solver.parameters.num_search_workers = min(8, self._cpu)

        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            return []

        Value = solver.Value
        return [[int(Value(v)) for v in start_vars[j]] for j in range(J)]