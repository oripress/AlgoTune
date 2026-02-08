from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        M = problem["num_machines"]
        jobs_data = problem["jobs"]
        J = len(jobs_data)

        if J == 0:
            return []

        total_ops = sum(len(job) for job in jobs_data)
        if total_ops == 0:
            return [[] for _ in range(J)]

        # Compute greedy upper bound with multiple dispatch rules
        greedy_starts, greedy_makespan = self._best_greedy(jobs_data, M)

        # Compute lower bound
        lb = self._lower_bound(jobs_data, M)

        # If lower bound equals upper bound, greedy is optimal
        if lb >= greedy_makespan:
            return greedy_starts

        horizon = greedy_makespan

        # For very small problems, just use CP-SAT directly
        # For larger problems, use tighter bounds and hints

        # Precompute prefix/suffix sums for tighter variable bounds
        prefix = []
        suffix = []
        for job in jobs_data:
            p = [0]
            for _, d in job:
                p.append(p[-1] + d)
            prefix.append(p)
            s = [0]
            for _, d in reversed(job):
                s.append(s[-1] + d)
            s.reverse()
            suffix.append(s)

        model = cp_model.CpModel()
        all_starts = {}
        all_ends = {}
        machine_to_intervals = [[] for _ in range(M)]

        for j in range(J):
            job = jobs_data[j]
            for k in range(len(job)):
                m, p = job[k]
                es = prefix[j][k]
                le = horizon - suffix[j][k + 1]
                start = model.new_int_var(es, le - p, f"s_{j}_{k}")
                end = model.new_int_var(es + p, le, f"e_{j}_{k}")
                interval = model.new_interval_var(start, p, end, f"i_{j}_{k}")
                all_starts[(j, k)] = start
                all_ends[(j, k)] = end
                machine_to_intervals[m].append(interval)
                if k > 0:
                    model.add(start >= all_ends[(j, k - 1)])
                model.add_hint(start, greedy_starts[j][k])

        for m_idx in range(M):
            if machine_to_intervals[m_idx]:
                model.add_no_overlap(machine_to_intervals[m_idx])

        makespan = model.new_int_var(lb, horizon, "makespan")
        last_ends = []
        for j in range(J):
            job = jobs_data[j]
            if job:
                last_ends.append(all_ends[(j, len(job) - 1)])
        if last_ends:
            model.add_max_equality(makespan, last_ends)
        model.minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.num_workers = 1
        solver.parameters.cp_model_presolve = True
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        solver.parameters.linearization_level = 0
        solver.parameters.use_lns_only = False
        
        status = solver.solve(model)

        if status == cp_model.OPTIMAL:
            solution = []
            for j in range(J):
                job = jobs_data[j]
                solution.append([int(solver.value(all_starts[(j, k)])) for k in range(len(job))])
            return solution

        return greedy_starts

    def _lower_bound(self, jobs_data, M):
        lb = 0
        machine_load = [0] * M
        for job in jobs_data:
            job_total = 0
            for m, p in job:
                job_total += p
                machine_load[m] += p
            if job_total > lb:
                lb = job_total
        for ml in machine_load:
            if ml > lb:
                lb = ml
        return lb

    def _best_greedy(self, jobs_data, M):
        best_ms = float('inf')
        best_st = None
        for rule in range(4):
            st = self._greedy(jobs_data, M, rule)
            ms = 0
            for j in range(len(jobs_data)):
                job = jobs_data[j]
                if job:
                    end = st[j][-1] + job[-1][1]
                    if end > ms:
                        ms = end
            if ms < best_ms:
                best_ms = ms
                best_st = st
        return best_st, best_ms

    def _greedy(self, jobs_data, M, rule):
        J = len(jobs_data)
        next_op = [0] * J
        job_ready = [0] * J
        machine_ready = [0] * M
        starts = [[] for _ in range(J)]

        remaining = []
        for job in jobs_data:
            nops = len(job)
            r = [0] * (nops + 1)
            for i in range(nops - 1, -1, -1):
                r[i] = r[i + 1] + job[i][1]
            remaining.append(r)

        total_ops = sum(len(job) for job in jobs_data)

        for _ in range(total_ops):
            best_key = None
            best_j = -1
            for j in range(J):
                k = next_op[j]
                if k >= len(jobs_data[j]):
                    continue
                m, p = jobs_data[j][k]
                jr = job_ready[j]
                mr = machine_ready[m]
                earliest = jr if jr > mr else mr

                if rule == 0:  # mwkr
                    key = (earliest, -remaining[j][k])
                elif rule == 1:  # spt
                    key = (earliest, p)
                elif rule == 2:  # lpt
                    key = (earliest, -p)
                else:  # fifo
                    key = (earliest, j)

                if best_key is None or key < best_key:
                    best_key = key
                    best_j = j

            j = best_j
            k = next_op[j]
            m, p = jobs_data[j][k]
            jr = job_ready[j]
            mr = machine_ready[m]
            s = jr if jr > mr else mr
            starts[j].append(s)
            job_ready[j] = s + p
            machine_ready[m] = s + p
            next_op[j] = k + 1

        return starts