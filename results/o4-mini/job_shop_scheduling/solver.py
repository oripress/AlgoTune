from typing import Any, List
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[List[int]]:
        # Parse string input if necessary
        if isinstance(problem, str):
            problem = eval(problem)
        M = problem["num_machines"]
        jobs_data = problem["jobs"]

        # Create the model
        model = cp_model.CpModel()
        # Compute horizon as sum of all durations
        horizon = sum(duration for job in jobs_data for _, duration in job)

        # Create tasks
        all_tasks = {}  # (job, op) -> (start, end, dur)
        machine_to_intervals = {m: [] for m in range(M)}
        for j, job in enumerate(jobs_data):
            for k, (machine, dur) in enumerate(job):
                suffix = f"_{j}_{k}"
                start_var = model.NewIntVar(0, horizon, f"start{suffix}")
                end_var = model.NewIntVar(0, horizon, f"end{suffix}")
                interval = model.NewIntervalVar(start_var, dur, end_var, f"interval{suffix}")
                all_tasks[(j, k)] = (start_var, end_var, dur)
                machine_to_intervals[machine].append(interval)
                # Precedence constraint
                if k > 0:
                    prev_end = all_tasks[(j, k-1)][1]
                    model.Add(start_var >= prev_end)

        # No overlap on machines
        for m in range(M):
            model.AddNoOverlap(machine_to_intervals[m])

        # Makespan
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ends = []
        for j, job in enumerate(jobs_data):
            _, end_var, _ = all_tasks[(j, len(job)-1)]
            last_ends.append(end_var)
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)

        # Extract solution
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution: List[List[int]] = []
            for j, job in enumerate(jobs_data):
                starts = []
                for k, _ in enumerate(job):
                    starts.append(int(solver.Value(all_tasks[(j, k)][0])))
                solution.append(starts)
            return solution
        # No solution found
        return []