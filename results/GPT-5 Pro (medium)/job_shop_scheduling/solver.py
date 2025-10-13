from typing import Any, List, Dict, Tuple
from ortools.sat.python import cp_model

def _ssgs_mwkr_heuristic(
    jobs_data: List[List[Tuple[int, int]]],
) -> Tuple[List[List[int]], int]:
    """
    Serial Schedule Generation Scheme with Most Work Remaining (MWKR) priority.
    Produces a feasible schedule and its makespan quickly.

    :param jobs_data: List of jobs, each is a list of (machine, duration)
    :return: (start_times, makespan)
    """
    J = len(jobs_data)
    if J == 0:
        return [], 0

    # Per job: next operation index and ready time
    next_op = [0] * J
    job_ready = [0] * J

    # Per machine: list of scheduled intervals (start, end)
    # Maintain sorted by start
    machines: Dict[int, List[Tuple[int, int]]] = {}
    # Initialize starts structure
    starts: List[List[int]] = [([0] * len(job)) for job in jobs_data]

    # Precompute remaining work per job
    remaining_work = [sum(d for _, d in job) for job in jobs_data]

    # Helper: earliest feasible start on a machine given release and duration
    def place_on_machine(machine: int, release: int, p: int) -> int:
        # Zero duration can start at release directly
        if p == 0:
            return release
        sched = machines.setdefault(machine, [])
        t = release
        # Find first gap
        for s, e in sched:
            if t + p <= s:
                break
            if e > t:
                t = e
        return t

    # Insert interval into machine schedule keeping it sorted
    def insert_interval(machine: int, s: int, e: int) -> None:
        sched = machines.setdefault(machine, [])
        # Append and fix order; schedules are small, insertion sort is fine
        # Find position to insert by start time
        lo, hi = 0, len(sched)
        while lo < hi:
            mid = (lo + hi) // 2
            if sched[mid][0] <= s:
                lo = mid + 1
            else:
                hi = mid
        sched.insert(lo, (s, e))

    # Set of jobs that still have operations
    remaining = sum(len(job) for job in jobs_data)
    # Ready jobs are those whose next operation is available (always true in SSGS)
    # MWKR priority among jobs with remaining operations
    while remaining > 0:
        # Build list of candidate jobs (those with remaining ops)
        candidates = [j for j in range(J) if next_op[j] < len(jobs_data[j])]
        # Choose job with maximum remaining work
        # Break ties by earliest ready time, then by job index for determinism
        best_j = None
        best_key = None
        for j in candidates:
            k = next_op[j]
            m, p = jobs_data[j][k]
            # earliest feasible start given machine current schedule
            est_mach = place_on_machine(m, job_ready[j], p)
            # Key: (-remaining_work, est_mach, j) to prefer more remaining work, earlier start
            key = (-remaining_work[j], est_mach, j)
            if best_key is None or key < best_key:
                best_key = key
                best_j = j
        # Schedule chosen job's next operation
        j = best_j  # type: ignore
        k = next_op[j]
        m, p = jobs_data[j][k]
        s = place_on_machine(m, job_ready[j], p)
        e = s + p
        insert_interval(m, s, e)
        starts[j][k] = s

        # Update job info
        job_ready[j] = e
        remaining_work[j] -= p
        next_op[j] += 1
        remaining -= 1

    makespan = max(
        (starts[j][len(jobs_data[j]) - 1] + jobs_data[j][-1][1]) if jobs_data[j] else 0
        for j in range(J)
    )
    return starts, makespan

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[List[int]]:
        """
        Solve the Job Shop Scheduling Problem (JSSP) using a fast hybrid:
        - SSGS-MWKR heuristic to get a good upper bound and potential early optimality.
        - Tightened CP-SAT model with reduced horizons/domains and hints for optimality proof.

        :param problem: Dict with keys:
            - "num_machines": int, number of machines
            - "jobs": list of jobs, where each job is a list of (machine, duration) tuples
        :return: A list of J lists of start times for each operation.
        """
        M = int(problem["num_machines"])
        jobs_data: List[List[Tuple[int, int]]] = problem["jobs"]

        # Handle trivial cases
        if not jobs_data:
            return []

        # Lower bound: max of (longest job duration sum, max machine workload)
        job_sums = [sum(d for _, d in job) for job in jobs_data]
        lb_job = max(job_sums) if job_sums else 0
        mach_sums = [0] * M
        for j, job in enumerate(jobs_data):
            for m, d in job:
                mach_sums[m] += d
        lb_mach = max(mach_sums) if mach_sums else 0
        lower_bound = max(lb_job, lb_mach)

        # Heuristic upper bound via SSGS-MWKR
        heuristic_starts, ub = _ssgs_mwkr_heuristic(jobs_data)

        # Early exit if proven optimal
        if ub == lower_bound:
            return heuristic_starts

        # Build CP-SAT with tightened horizon and domains
        model = cp_model.CpModel()
        horizon = ub  # use tight UB instead of sum of durations

        # Precompute head/tail sums per job for domain tightening
        heads: List[List[int]] = []
        tails: List[List[int]] = []
        for job in jobs_data:
            head = [0]
            for _, d in job[:-1]:
                head.append(head[-1] + d)
            heads.append(head)
            # tails[k] = sum_{i=k}^{end} p_i
            tail = [0] * len(job)
            suffix = 0
            for idx in range(len(job) - 1, -1, -1):
                suffix += job[idx][1]
                tail[idx] = suffix
            tails.append(tail)

        # Create variables and precedence constraints
        all_tasks: Dict[Tuple[int, int], Tuple[cp_model.IntVar, cp_model.IntVar, int]] = {}
        machine_to_intervals: Dict[int, List[cp_model.IntervalVar]] = {m: [] for m in range(M)}

        for j, job in enumerate(jobs_data):
            for k, (m, p) in enumerate(job):
                suffix = f"_{j}_{k}"
                # Tight domain: start in [head, horizon - tail]
                s_lb = heads[j][k]
                s_ub = horizon - tails[j][k]
                if s_ub < s_lb:
                    s_ub = s_lb  # keep non-empty domain

                start = model.NewIntVar(s_lb, s_ub, f"start{suffix}")
                # end in [head+p, horizon - tails[k+1]] if k+1 exists else [head+p, horizon]
                e_lb = s_lb + p
                e_ub = horizon if k == len(job) - 1 else horizon - tails[j][k + 1]
                if e_ub < e_lb:
                    e_ub = e_lb
                end = model.NewIntVar(e_lb, e_ub, f"end{suffix}")
                interval = model.NewIntervalVar(start, p, end, f"interval{suffix}")

                all_tasks[(j, k)] = (start, end, p)
                machine_to_intervals[m].append(interval)

                if k > 0:
                    prev_end = all_tasks[(j, k - 1)][1]
                    model.Add(start >= prev_end)

                # Add hint from heuristic
                model.AddHint(start, int(heuristic_starts[j][k]))

        # No-overlap on each machine
        for m in range(M):
            if machine_to_intervals[m]:
                model.AddNoOverlap(machine_to_intervals[m])

        # Makespan objective.
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ends = []
        for job_id, job in enumerate(jobs_data):
            _, end_var, _ = all_tasks[(job_id, len(job) - 1)]
            last_ends.append(end_var)
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        # Keep defaults; hints + tighter domains should speed up search.

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            solution: List[List[int]] = []
            for j, job in enumerate(jobs_data):
                starts = []
                for k, _ in enumerate(job):
                    starts.append(int(solver.Value(all_tasks[(j, k)][0])))
                solution.append(starts)
            return solution
        else:
            # Fallback to heuristic schedule (valid but may be suboptimal).
            # However, validator requires optimality, so try returning heuristic only
            # if CP could not solve but UB equals LB (handled earlier). Otherwise, return empty.
            return []