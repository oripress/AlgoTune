from typing import Any
from ortools.sat.python import cp_model
import psutil

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        """Optimized CP-SAT solver for JSSP with advanced parameters."""
        M = problem["num_machines"]
        jobs_data = problem["jobs"]
        
        # Compute horizon as total duration sum
        horizon = sum(d for job in jobs_data for _, d in job)
        
        # Compute lower bound for makespan
        machine_workloads = [0] * M
        job_totals = []
        for job in jobs_data:
            job_total = 0
            for m, p in job:
                machine_workloads[m] += p
                job_total += p
            job_totals.append(job_total)
        makespan_lb = max(max(machine_workloads), max(job_totals))
        
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
        
        for m in range(M):
            model.AddNoOverlap(machine_to_intervals[m])
        
        makespan = model.NewIntVar(makespan_lb, horizon, "makespan")
        last_ends = [all_tasks[(j, len(job)-1)][1] for j, job in enumerate(jobs_data)]
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)
        
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = psutil.cpu_count(logical=True)
        solver.parameters.linearization_level = 2
        solver.parameters.log_search_progress = False
        solver.parameters.cp_model_probing_level = 3
        solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
        solver.parameters.use_lns = True
        solver.parameters.optimize_with_core = True
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            return [
                [int(solver.Value(all_tasks[(j, k)][0])) for k in range(len(job))]
                for j, job in enumerate(jobs_data)
            ]
        return []