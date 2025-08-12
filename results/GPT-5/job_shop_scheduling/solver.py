from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[List[int]]:
        """
        Solve the Job Shop Scheduling Problem (JSSP) optimally using CP-SAT,
        but accelerate with a fast constructive heuristic to tighten bounds
        and provide solution hints.

        Input:
          problem = {
              "num_machines": M,
              "jobs": [
                  [(machine_0, duration_0), (machine_1, duration_1), ...],
                  ...
              ]
          }

        Output:
          A list of J lists of start times for each operation in each job.
        """
        M: int = int(problem["num_machines"])
        jobs: List[List[Tuple[int, int]]] = problem["jobs"]
        J: int = len(jobs)

        # Trivial cases
        if J == 0:
            return []
        if M == 0:
            # No machines: every job must be empty
            return [[] for _ in range(J)]

        # Precompute per-job durations, totals, and counts
        job_lengths: List[int] = [len(job) for job in jobs]
        total_ops: int = sum(job_lengths)
        if total_ops == 0:
            return [[] for _ in range(J)]

        # Special-case: single job => precedence chain only; trivial optimal schedule.
        if J == 1:
            starts = [[0] * job_lengths[0]]
            t = 0
            for k, (_, p) in enumerate(jobs[0]):
                starts[0][k] = t
                t += p
            return starts

        # Special-case: single machine => schedule jobs one after another; optimal makespan = sum durations.
        if M == 1:
            t = 0
            starts: List[List[int]] = [[] for _ in range(J)]
            for j, job in enumerate(jobs):
                for _, p in job:
                    starts[j].append(t)
                    t += p
            return starts

        # Compute horizon = sum of all durations (safe upper bound)
        all_durations_sum = sum(d for job in jobs for _, d in job)

        # Lower bound on makespan: max of job duration sums and machine loads
        job_sums = [sum(d for _, d in job) for job in jobs]
        machine_sums = [0] * M
        for job in jobs:
            for m, d in job:
                machine_sums[m] += d
        LB: int = max(max(job_sums) if job_sums else 0, max(machine_sums) if machine_sums else 0)

        # Heuristic schedule to get a strong UB and hints
        heuristic_starts, heuristic_makespan = self._heuristic_schedule_gt(jobs, M)

        # Use the heuristic UB if it exists; otherwise fall back to sum durations
        UB: int = heuristic_makespan if heuristic_makespan is not None else all_durations_sum
        if UB < 0:
            UB = all_durations_sum

        # If heuristic reaches the lower bound, it is optimal. Return immediately.
        if heuristic_starts is not None and heuristic_makespan == LB:
            return heuristic_starts

        # Precompute per-op earliest and latest starts using only job chains and UB
        # earliest_start[j][k] = sum durations of job j before op k
        # latest_start[j][k] = UB - sum durations of job j from op k to end (inclusive)
        earliest_start: List[List[int]] = []
        latest_start: List[List[int]] = []
        for j, job in enumerate(jobs):
            pj = [d for _, d in job]
            pref = [0]
            for d in pj:
                pref.append(pref[-1] + d)
            total = pref[-1]
            est_list = pref[:-1]  # k -> sum of first k durations
            # tail including current op k: total - pref[k]
            lst_list = [UB - (total - pref[k]) for k in range(len(job))]
            # Ensure bounds are within [0, UB]
            est_list = [max(0, min(UB, v)) for v in est_list]
            lst_list = [max(0, min(UB, v)) for v in lst_list]
            earliest_start.append(est_list)
            latest_start.append(lst_list)

        # Build CP-SAT model with tightened bounds and hints
        model = cp_model.CpModel()

        # Create variables and intervals
        # We'll cap end variables at UB as makespan will be <= UB.
        all_tasks: Dict[Tuple[int, int], Tuple[cp_model.IntVar, cp_model.IntVar, int]] = {}
        machine_to_intervals: List[List[cp_model.IntervalVar]] = [[] for _ in range(M)]

        for j, job in enumerate(jobs):
            for k, (m, p) in enumerate(job):
                # Tight start bounds
                lb = int(earliest_start[j][k])
                ub = int(latest_start[j][k])
                if ub < lb:
                    # In rare degenerate cases, relax to [0, UB] to keep model consistent.
                    lb, ub = 0, UB
                start = model.NewIntVar(lb, ub, f"start_{j}_{k}")
                end = model.NewIntVar(0, UB, f"end_{j}_{k}")
                interval = model.NewIntervalVar(start, p, end, f"interval_{j}_{k}")
                all_tasks[(j, k)] = (start, end, p)
                machine_to_intervals[m].append(interval)

                # Chain precedence within job
                if k > 0:
                    prev_end = all_tasks[(j, k - 1)][1]
                    model.Add(start >= prev_end)

        # No-overlap per machine
        for m in range(M):
            if machine_to_intervals[m]:
                model.AddNoOverlap(machine_to_intervals[m])

        # Makespan objective bounded by [LB, UB]
        makespan = model.NewIntVar(LB, UB, "makespan")
        last_ends: List[cp_model.IntVar] = []
        for j, job in enumerate(jobs):
            if job:
                _, end_var, _ = all_tasks[(j, len(job) - 1)]
                last_ends.append(end_var)
        if last_ends:
            model.AddMaxEquality(makespan, last_ends)
        else:
            # No operations at all
            model.Add(makespan == 0)

        model.Minimize(makespan)

        # Add solution hints from heuristic schedule
        if heuristic_starts is not None:
            for j, job in enumerate(jobs):
                for k, (_, p) in enumerate(job):
                    s_hint = int(heuristic_starts[j][k])
                    model.AddHint(all_tasks[(j, k)][0], s_hint)
                    model.AddHint(all_tasks[(j, k)][1], s_hint + p)
            model.AddHint(makespan, int(UB))

        # Solve with multiple search workers for speed
        solver = cp_model.CpSolver()
        try:
            import os

            workers = os.cpu_count() or 1
            # Avoid oversubscription; cap at 8 which generally provides good performance.
            solver.parameters.num_search_workers = min(8, max(1, workers))
        except Exception:
            solver.parameters.num_search_workers = 8

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            solution: List[List[int]] = []
            for j, job in enumerate(jobs):
                starts_j: List[int] = [0] * len(job)
                for k, _ in enumerate(job):
                    starts_j[k] = int(solver.Value(all_tasks[(j, k)][0]))
                solution.append(starts_j)
            return solution

        # Fallback: if somehow not optimal, return heuristic schedule (feasible).
        if heuristic_starts is not None:
            return heuristic_starts
        return [[] for _ in range(J)]

    @staticmethod
    def _heuristic_schedule_gt(
        jobs: List[List[Tuple[int, int]]], M: int
    ) -> Tuple[List[List[int]] | None, int | None]:
        """
        Giffler-Thompson non-delay heuristic for Job Shop Scheduling.
        At each step:
          - Among all available ops (next op of each job), compute earliest start s and completion c=s+p.
          - Select op with smallest c; identify its machine m*.
          - Build conflict set of available ops requiring m* with s_i < c*.
          - Schedule one op from the conflict set according to tie-breaking priority.
          - Repeat until all operations are scheduled.

        Tie-breaking within conflict set (and when it is empty, schedule the chosen op):
          1) minimal earliest start time s
          2) minimal earliest completion time c
          3) maximal remaining work (sum of remaining durations in the job starting at this op)
          4) maximal processing time p
          5) lowest job index

        Returns (starts, makespan) or (None, None) on failure.
        """
        J = len(jobs)
        if J == 0:
            return [], 0
        job_len = [len(job) for job in jobs]
        if sum(job_len) == 0:
            return [[] for _ in range(J)], 0

        job_ready = [0] * J  # ready time for next op in each job
        machine_ready = [0] * M  # ready time for each machine
        starts: List[List[int]] = [[0] * job_len[j] for j in range(J)]
        next_op = [0] * J

        # Precompute remaining work per job position (tails)
        rem_work: List[List[int]] = []
        for j, job in enumerate(jobs):
            pj = [d for _, d in job]
            tail = [0] * len(job)
            acc = 0
            for k in range(len(job) - 1, -1, -1):
                acc += pj[k]
                tail[k] = acc
            rem_work.append(tail)

        remaining = sum(job_len)
        while remaining > 0:
            # Build list of available ops and their earliest start and completion times
            avail: List[Tuple[int, int, int, int, int, int]] = []  # (j, k, m, p, s, c)
            for j in range(J):
                k = next_op[j]
                if k >= job_len[j]:
                    continue
                m, p = jobs[j][k]
                s = job_ready[j] if job_ready[j] >= machine_ready[m] else machine_ready[m]
                c = s + p
                avail.append((j, k, m, p, s, c))

            if not avail:
                # Should not happen
                return None, None

            # Pick operation with minimal completion time
            j_star, k_star, m_star, p_star, s_star, c_star = min(avail, key=lambda t: (t[5], t[4], -rem_work[t[0]][t[1]], -t[3], t[0]))

            # Conflict set: ops on same machine with s_i < c_star
            conflict: List[Tuple[int, int, int, int, int, int]] = [t for t in avail if t[2] == m_star and t[4] < c_star]

            # If conflict set is empty (can happen when p_star = 0), schedule the selected op directly.
            if not conflict:
                conflict = [(j_star, k_star, m_star, p_star, s_star, c_star)]

            # Choose from conflict set by tie-breaking
            j, k, m, p, s, c = min(
                conflict, key=lambda t: (t[4], t[5], -rem_work[t[0]][t[1]], -t[3], t[0])
            )

            # Schedule chosen operation
            starts[j][k] = s
            finish = s + p
            job_ready[j] = finish
            machine_ready[m] = finish
            next_op[j] += 1
            remaining -= 1

        makespan = max(job_ready) if job_ready else 0
        return starts, makespan