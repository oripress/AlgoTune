import collections
from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        """
        Solve the JSSP using CP-SAT with several optimizations:
        1. Tighter variable domains using pre-calculated heads and tails.
        2. Tighter makespan lower bound (max of machine load and job length).
        3. Parallel search workers.
        """
        M = problem["num_machines"]
        jobs_data = problem["jobs"]

        if not jobs_data:
            return []

        # --- Model Enhancement: Tighter bounds ---
        # A simple upper bound for the makespan (sum of all durations).
        horizon = sum(p for job in jobs_data for _, p in job)
        
        # Calculate a lower bound for the makespan.
        machine_loads = [0] * M
        for job in jobs_data:
            for m, p in job:
                machine_loads[m] += p
        job_lengths = [sum(p for _, p in job) for job in jobs_data]
        
        min_makespan_lower_bound = 0
        if machine_loads:
            min_makespan_lower_bound = max(min_makespan_lower_bound, max(machine_loads))
        if job_lengths:
            min_makespan_lower_bound = max(min_makespan_lower_bound, max(job_lengths))

        # Pre-calculate heads (earliest start times) and tails (time to complete from start) for each task.
        job_heads = []
        job_tails = []
        for job in jobs_data:
            # heads: sum of durations of preceding tasks
            head = 0
            heads = []
            for _, p in job:
                heads.append(head)
                head += p
            job_heads.append(heads)
            
            # tails: sum of durations of succeeding tasks
            tail = 0
            tails = []
            for _, p in reversed(job):
                tails.append(tail)
                tail += p
            job_tails.append(list(reversed(tails)))

        model = cp_model.CpModel()
        
        # --- Model Creation ---
        all_tasks = {}  # (j,k) -> (start_var, end_var, interval_var)
        machine_to_intervals: dict[int, list[cp_model.IntervalVar]] = collections.defaultdict(list)
        
        for j, job in enumerate(jobs_data):
            for k, (m, p) in enumerate(job):
                head_jk = job_heads[j][k]
                tail_jk = job_tails[j][k]
                
                # Tighter domains for start and end variables
                start_var = model.NewIntVar(head_jk, horizon - tail_jk - p, f"start_{j}_{k}")
                end_var = model.NewIntVar(head_jk + p, horizon - tail_jk, f"end_{j}_{k}")
                
                interval = model.NewIntervalVar(start_var, p, end_var, f"interval_{j}_{k}")
                all_tasks[(j, k)] = (start_var, end_var, interval)
                machine_to_intervals[m].append(interval)
                
                # Precedence constraint: operation k must follow k-1
                if k > 0:
                    prev_end = all_tasks[(j, k - 1)][1]
                    model.Add(start_var >= prev_end)

        # No-overlap constraint on each machine
        for m in range(M):
            model.AddNoOverlap(machine_to_intervals[m])

        # Makespan objective
        makespan = model.NewIntVar(min_makespan_lower_bound, horizon, "makespan")
        last_ends = [all_tasks[(j, len(job) - 1)][1] for j, job in enumerate(jobs_data) if job]
        if last_ends:
            model.AddMaxEquality(makespan, last_ends)
        else:
            model.Add(makespan == 0)
        model.Minimize(makespan)

        # --- Solver Configuration and Execution ---
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        
        status = solver.Solve(model)

        # --- Solution Extraction ---
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = []
            for j, job in enumerate(jobs_data):
                starts = [int(solver.Value(all_tasks[(j, k)][0])) for k in range(len(job))]
                solution.append(starts)
            return solution
        else:
            return []