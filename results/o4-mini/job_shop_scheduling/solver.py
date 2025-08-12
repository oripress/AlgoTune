import multiprocessing
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        M = problem["num_machines"]
        jobs_data = problem["jobs"]
        # Compute lower bound from job and machine workloads
        job_sums = [sum(p for (_, p) in job) for job in jobs_data]
        machine_sums = [0] * M
        for job in jobs_data:
            for m, p in job:
                machine_sums[m] += p
        lower_bound = max(max(job_sums), max(machine_sums))
        # Quick greedy heuristic for initial solution
        greedy_sched = self._greedy_schedule(M, jobs_data)
        # Compute greedy makespan
        greedy_makespan = max(
            greedy_sched[j][k] + p
            for j, job in enumerate(jobs_data)
            for k, (_, p) in enumerate(job)
        )
        if greedy_makespan == lower_bound:
            return greedy_sched
        # Use greedy makespan as horizon
        horizon = greedy_makespan
        # Compute earliest and latest start times to prune
        prefix_sums = []
        rem_sums = []
        for job in jobs_data:
            ps = [0]
            for _, p in job:
                ps.append(ps[-1] + p)
            total = ps[-1]
            prefix_sums.append(ps[:-1])
            rem_sums.append([total - ps[i+1] for i in range(len(job))])
        # Build CP-SAT model
        model = cp_model.CpModel()

        all_tasks = {}
        machine_to_intervals = {m: [] for m in range(M)}

        for j, job in enumerate(jobs_data):
            for k, (m, p) in enumerate(job):
                suffix = f"_{j}_{k}"
                start = model.NewIntVar(0, horizon, f"start{suffix}")
                end = model.NewIntVar(0, horizon, f"end{suffix}")
                interval = model.NewIntervalVar(start, p, end, f"interval{suffix}")
                all_tasks[(j, k)] = (start, end, p)
                machine_to_intervals[m].append(interval)
                if k > 0:
                    prev_end = all_tasks[(j, k - 1)][1]
                    model.Add(start >= prev_end)

        # No overlap on machines
        for m in range(M):
            model.AddNoOverlap(machine_to_intervals[m])

        # Makespan objective
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ends = []
        for j, job in enumerate(jobs_data):
            _, end_var, _ = all_tasks[(j, len(job) - 1)]
            last_ends.append(end_var)
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)

        # Provide hints from greedy solution
        try:
            for j, job in enumerate(jobs_data):
                for k, (m, p) in enumerate(job):
                    start_var, end_var, _ = all_tasks[(j, k)]
                    hint_s = greedy_sched[j][k]
                    model.AddHint(start_var, hint_s)
                    model.AddHint(end_var, hint_s + p)
            model.AddHint(makespan, greedy_makespan)
        except Exception:
            pass

        # Decision strategy: schedule tasks with larger remaining work first
        flat_list = [(rem_sums[j][k], all_tasks[(j, k)][0])
                     for j, job in enumerate(jobs_data)
                     for k, _ in enumerate(job)]
        flat_list.sort(reverse=True, key=lambda x: x[0])
        decision_vars = [var for _, var in flat_list]
        model.AddDecisionStrategy(decision_vars,
                                   cp_model.CHOOSE_FIRST,
                                   cp_model.SELECT_MIN_VALUE)

        # Solve with CP-SAT
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        status = solver.Solve(model)
        result = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for j, job in enumerate(jobs_data):
                starts = []
                for k, _ in enumerate(job):
                    starts.append(int(solver.Value(all_tasks[(j, k)][0])))
                result.append(starts)
        return result

    def _greedy_schedule(self, M, jobs_data):
        J = len(jobs_data)
        next_op = [0] * J
        job_end = [0] * J
        machine_end = [0] * M
        schedule = [ [0] * len(job) for job in jobs_data ]
        total_ops = sum(len(job) for job in jobs_data)
        for _ in range(total_ops):
            best_j = -1
            best_start = None
            best_p = None
            for j in range(J):
                k = next_op[j]
                if k >= len(jobs_data[j]):
                    continue
                m, p = jobs_data[j][k]
                est = max(job_end[j], machine_end[m])
                if best_j == -1 or est < best_start or (est == best_start and p > best_p):
                    best_j = j
                    best_start = est
                    best_p = p
            k = next_op[best_j]
            m, p = jobs_data[best_j][k]
            schedule[best_j][k] = best_start
            job_end[best_j] = best_start + p
            machine_end[m] = best_start + p
            next_op[best_j] += 1
        return schedule