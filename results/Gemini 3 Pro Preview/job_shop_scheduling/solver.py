import psutil
from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        """
        Solve the JSSP using CP-SAT with interval variables and no-overlap.
        Optimized with multi-threading.
        """
        M = problem["num_machines"]
        jobs_data = problem["jobs"]

        model = cp_model.CpModel()
        
        # Compute horizon - sum of all durations is a safe upper bound
        horizon = sum(d for job in jobs_data for _, d in job)

        # Create interval vars and precedence constraints
        all_tasks = {}  # (j,k) -> (start_var, end_var, duration)
        machine_to_intervals = {m: [] for m in range(M)}
        
        for j, job in enumerate(jobs_data):
            for k, (m, p) in enumerate(job):
                # Optimization: Don't name variables to save string construction time
                start = model.NewIntVar(0, horizon, "")
                end = model.NewIntVar(0, horizon, "")
                interval = model.NewIntervalVar(start, p, end, "")
                all_tasks[(j, k)] = (start, end, p)
                machine_to_intervals[m].append(interval)
                if k > 0:
                    prev_end = all_tasks[(j, k - 1)][1]
                    model.Add(start >= prev_end)

        # No-overlap on each machine
        for m in range(M):
            model.AddNoOverlap(machine_to_intervals[m])

        # Makespan objective.
        makespan = model.NewIntVar(0, horizon, "")
        last_ends = []
        for job_id, job in enumerate(jobs_data):
            _, end_var, _ = all_tasks[(job_id, len(job) - 1)]
            last_ends.append(end_var)
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        # Optimization: Use all available cores
        solver.parameters.num_search_workers = psutil.cpu_count(logical=True)
        
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            solution = []
            for j, job in enumerate(jobs_data):
                starts = []
                for k, _ in enumerate(job):
                    starts.append(int(solver.Value(all_tasks[(j, k)][0])))
                solution.append(starts)
            return solution
        else:
            return []