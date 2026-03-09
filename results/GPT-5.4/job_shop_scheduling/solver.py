from __future__ import annotations

from typing import Any

from ortools.sat.python import cp_model

class Solver:
    def _build_data(self, problem: dict[str, Any]):
        jobs = problem["jobs"]
        num_machines = int(problem["num_machines"])
        num_jobs = len(jobs)

        job_lens = [len(job) for job in jobs]
        total_ops = sum(job_lens)

        machines = []
        durations = []
        prefix = []
        rem_job = []
        tail_excl = []
        machine_loads = [0] * num_machines

        for job in jobs:
            jm = [m for m, _ in job]
            jp = [p for _, p in job]
            machines.append(jm)
            durations.append(jp)

            pref = [0] * (len(job) + 1)
            for i, p in enumerate(jp):
                pref[i + 1] = pref[i] + p
                machine_loads[jm[i]] += p
            prefix.append(pref)

            rem = [0] * (len(job) + 1)
            tail = [0] * (len(job) + 1)
            s = 0
            for i in range(len(job) - 1, -1, -1):
                s += jp[i]
                rem[i] = s
                tail[i] = s - jp[i]
            rem_job.append(rem)
            tail_excl.append(tail)

        return {
            "jobs": jobs,
            "num_machines": num_machines,
            "num_jobs": num_jobs,
            "job_lens": job_lens,
            "total_ops": total_ops,
            "machines": machines,
            "durations": durations,
            "prefix": prefix,
            "rem_job": rem_job,
            "tail_excl": tail_excl,
            "machine_loads": machine_loads,
        }

    def _solve_single_job(self, durations: list[list[int]]) -> list[list[int]]:
        out = []
        for jp in durations:
            t = 0
            starts = []
            for p in jp:
                starts.append(t)
                t += p
            out.append(starts)
        return out

    def _solve_single_machine(
        self,
        num_jobs: int,
        job_lens: list[int],
        durations: list[list[int]],
    ) -> list[list[int]]:
        sol = [[0] * job_lens[j] for j in range(num_jobs)]
        t = 0
        for j in range(num_jobs):
            for k, p in enumerate(durations[j]):
                sol[j][k] = t
                t += p
        return sol

    def _solve_unit_jobs(
        self,
        num_jobs: int,
        num_machines: int,
        jobs: list[list[tuple[int, int]]],
    ) -> list[list[int]]:
        per_machine: list[list[int]] = [[] for _ in range(num_machines)]
        for j, job in enumerate(jobs):
            if job:
                per_machine[job[0][0]].append(j)

        sol = [[0] * len(job) for job in jobs]
        for m in range(num_machines):
            t = 0
            for j in per_machine[m]:
                sol[j][0] = t
                t += jobs[j][0][1]
        return sol

    def _heuristic_schedule(
        self,
        num_jobs: int,
        num_machines: int,
        job_lens: list[int],
        machines: list[list[int]],
        durations: list[list[int]],
        rem_job: list[list[int]],
        rule: int,
    ) -> tuple[int, list[list[int]]]:
        next_idx = [0] * num_jobs
        job_ready = [0] * num_jobs
        machine_ready = [0] * num_machines
        starts = [[0] * job_lens[j] for j in range(num_jobs)]

        total_ops = sum(job_lens)
        done = 0
        makespan = 0
        while done < total_ops:
            frontier = []
            ec_min = None
            min_machines = set()

            for j in range(num_jobs):
                k = next_idx[j]
                if k >= job_lens[j]:
                    continue
                m = machines[j][k]
                p = durations[j][k]
                est = job_ready[j]
                mr = machine_ready[m]
                if mr > est:
                    est = mr
                ec = est + p
                frontier.append((j, k, m, p, est, ec))
                if ec_min is None or ec < ec_min:
                    ec_min = ec

            for _, _, m, _, _, ec in frontier:
                if ec == ec_min:
                    min_machines.add(m)

            candidates = [
                item
                for item in frontier
                if item[2] in min_machines and (item[4] < ec_min or item[5] == ec_min)
            ]
            if not candidates:
                candidates = frontier

            def score(item: tuple[int, int, int, int, int, int]):
                j, k, _m, p, est, ec = item
                remaining = rem_job[j][k]
                if rule == 0:
                    return (ec, est, p, -remaining, j)
                if rule == 1:
                    return (ec, -remaining, p, est, j)
                if rule == 2:
                    return (est, p, ec, -remaining, j)
                return (est, -remaining, ec, -p, j)

            j, k, m, p, est, _ = min(candidates, key=score)
            starts[j][k] = est
            end = est + p
            job_ready[j] = end
            machine_ready[m] = end
            if end > makespan:
                makespan = end
            next_idx[j] = k + 1
            done += 1

        return makespan, starts

    def _cp_sat_solve(
        self,
        num_jobs: int,
        num_machines: int,
        job_lens: list[int],
        machines: list[list[int]],
        durations: list[list[int]],
        prefix: list[list[int]],
        tail_excl: list[list[int]],
        lower_bound: int,
        upper_bound: int,
        hint: list[list[int]],
    ) -> list[list[int]]:
        model = cp_model.CpModel()

        intervals_by_machine: list[list[Any]] = [[] for _ in range(num_machines)]
        start_vars: list[list[Any]] = []
        job_ends: list[Any] = []
        fixed_ctor = getattr(model, "NewFixedSizeIntervalVar", None)
        if fixed_ctor is None:
            fixed_ctor = getattr(model, "new_fixed_size_interval_var", None)

        for j in range(num_jobs):
            row = []
            prev_end = None
            for k in range(job_lens[j]):
                p = durations[j][k]
                head = prefix[j][k]
                latest_start = upper_bound - p - tail_excl[j][k]
                if latest_start < 0:
                    latest_start = 0

                start = model.NewIntVar(0, latest_start, "")
                if head > 0:
                    model.Add(start >= head)

                if fixed_ctor is not None:
                    interval = fixed_ctor(start, p, "")
                    end_expr = start + p
                else:
                    end = model.NewIntVar(p, upper_bound, "")
                    interval = model.NewIntervalVar(start, p, end, "")
                    end_expr = end

                intervals_by_machine[machines[j][k]].append(interval)
                if prev_end is not None:
                    model.Add(start >= prev_end)
                prev_end = end_expr
                row.append(start)
                model.AddHint(start, hint[j][k])

            start_vars.append(row)
            if prev_end is not None:
                job_ends.append(prev_end)

        for m in range(num_machines):
            if len(intervals_by_machine[m]) > 1:
                model.AddNoOverlap(intervals_by_machine[m])

        if not job_ends:
            return [[] for _ in range(num_jobs)]

        makespan = model.NewIntVar(lower_bound, upper_bound, "")
        for end_expr in job_ends:
            model.Add(makespan >= end_expr)
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        params = solver.parameters
        params.num_search_workers = 1
        if hasattr(params, "use_optimization_hints"):
            params.use_optimization_hints = True
        if hasattr(params, "linearization_level"):
            params.linearization_level = 0

        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            return hint

        return [
            [solver.Value(start_vars[j][k]) for k in range(job_lens[j])]
            for j in range(num_jobs)
        ]

    def solve(self, problem, **kwargs) -> Any:
        data = self._build_data(problem)
        jobs = data["jobs"]
        num_machines = data["num_machines"]
        num_jobs = data["num_jobs"]
        job_lens = data["job_lens"]
        total_ops = data["total_ops"]
        machines = data["machines"]
        durations = data["durations"]
        prefix = data["prefix"]
        rem_job = data["rem_job"]
        tail_excl = data["tail_excl"]
        machine_loads = data["machine_loads"]

        if total_ops == 0:
            return [[] for _ in range(num_jobs)]

        if num_jobs <= 1:
            return self._solve_single_job(durations)

        if num_machines <= 1:
            return self._solve_single_machine(num_jobs, job_lens, durations)

        if max(job_lens) <= 1:
            return self._solve_unit_jobs(num_jobs, num_machines, jobs)

        machine_owner = [-1] * num_machines
        independent = True
        for j in range(num_jobs):
            for m in machines[j]:
                owner = machine_owner[m]
                if owner == -1:
                    machine_owner[m] = j
                elif owner != j:
                    independent = False
                    break
            if not independent:
                break
        if independent:
            return self._solve_single_job(durations)

        lower_bound = 0
        for j in range(num_jobs):
            v = rem_job[j][0]
            if v > lower_bound:
                lower_bound = v
        for m in range(num_machines):
            v = machine_loads[m]
            if v > lower_bound:
                lower_bound = v

        heuristic_best = None
        heuristic_sol = None
        for rule in (1, 0, 2, 3):
            ms, sol = self._heuristic_schedule(
                num_jobs,
                num_machines,
                job_lens,
                machines,
                durations,
                rem_job,
                rule,
            )
            if heuristic_best is None or ms < heuristic_best:
                heuristic_best = ms
                heuristic_sol = sol
                if heuristic_best == lower_bound:
                    break

        if heuristic_best == lower_bound:
            return heuristic_sol

        return self._cp_sat_solve(
            num_jobs,
            num_machines,
            job_lens,
            machines,
            durations,
            prefix,
            tail_excl,
            lower_bound,
            heuristic_best,
            heuristic_sol,
        )