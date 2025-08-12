from ortools.sat.python import cp_model
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[list[int]]:
        """Solve JSSP using OR-Tools CP-SAT with optimizations."""
        M = problem["num_machines"]
        jobs_data = problem["jobs"]
        
        model = cp_model.CpModel()
        
        # Compute horizon
        horizon = sum(d for job in jobs_data for _, d in job)
        
        # Create variables
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
                
                # Precedence constraints
                if k > 0:
                    prev_end = all_tasks[(j, k - 1)][1]
                    model.Add(start >= prev_end)
        
        # No-overlap constraints
        for m in range(M):
            model.AddNoOverlap(machine_to_intervals[m])
        
        # Minimize makespan
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ends = []
        for j, job in enumerate(jobs_data):
            _, end_var, _ = all_tasks[(j, len(job) - 1)]
            last_ends.append(end_var)
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)
        
        # Solve with optimized parameters
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 1.0
        solver.parameters.num_search_workers = 4
        status = solver.Solve(model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            solution = []
            for j, job in enumerate(jobs_data):
                starts = []
                for k, _ in enumerate(job):
                    starts.append(int(solver.Value(all_tasks[(j, k)][0])))
                solution.append(starts)
            return solution
        else:
            return []