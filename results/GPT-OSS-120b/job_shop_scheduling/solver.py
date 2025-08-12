from typing import Any, List
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[List[int]]:
        """
        Solve the Job Shop Scheduling Problem using OR-Tools CP-SAT.

        Returns a list of start times for each operation in each job.
        """
        num_machines = problem["num_machines"]
        jobs = problem["jobs"]

        # Compute a simple horizon (upper bound on makespan)
        horizon = sum(duration for job in jobs for _, duration in job)

        model = cp_model.CpModel()

        # Create variables
        all_tasks = {}  # (job, op) -> (start, end, interval)
        machine_to_intervals: dict[int, List[cp_model.IntervalVar]] = {m: [] for m in range(num_machines)}

        for j, job in enumerate(jobs):
            for k, (machine, duration) in enumerate(job):
                suffix = f"_{j}_{k}"
                start = model.NewIntVar(0, horizon, f"start{suffix}")
                end = model.NewIntVar(0, horizon, f"end{suffix}")
                interval = model.NewIntervalVar(start, duration, end, f"interval{suffix}")
                all_tasks[(j, k)] = (start, end, interval)
                machine_to_intervals[machine].append(interval)

                # Precedence constraints within a job
                if k > 0:
                    prev_end = all_tasks[(j, k - 1)][1]
                    model.Add(start >= prev_end)

        # No-overlap constraints for each machine
        # No-overlap constraints for each machine
        for m in range(num_machines):
            model.AddNoOverlap(machine_to_intervals[m])

        # Add greedy schedule hints to speed up solving.
        machine_available = {m: 0 for m in range(num_machines)}
        job_last_end = {j: 0 for j in range(len(jobs))}
        for j, job in enumerate(jobs):
            for k, (machine, duration) in enumerate(job):
                start_var = all_tasks[(j, k)][0]
                est = max(machine_available[machine], job_last_end[j])
                model.AddHint(start_var, est)
                machine_available[machine] = est + duration
                job_last_end[j] = est + duration
        # Makespan variable
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ends = [all_tasks[(j, len(job) - 1)][1] for j, job in enumerate(jobs)]
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        # Use parallel workers for faster convergence and a shorter time limit.
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = 0.5

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            result: List[List[int]] = []
            for j, job in enumerate(jobs):
                starts: List[int] = []
                for k in range(len(job)):
                    starts.append(solver.Value(all_tasks[(j, k)][0]))
                result.append(starts)
            return result
        # No optimal solution found
        return []