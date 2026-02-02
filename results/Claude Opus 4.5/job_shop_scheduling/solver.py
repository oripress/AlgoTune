from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[list[int]]:
        M = problem["num_machines"]
        jobs_data = problem["jobs"]
        
        if not jobs_data:
            return []
        
        J = len(jobs_data)
        
        # Compute tighter bounds
        # Early start times (minimum possible start based on job precedence)
        early_start = []
        job_durations = []
        for job in jobs_data:
            es = []
            cumsum = 0
            for m, p in job:
                es.append(cumsum)
                cumsum += p
            early_start.append(es)
            job_durations.append(cumsum)
        
        # Compute machine loads and lower bound
        machine_loads = [0] * M
        for job in jobs_data:
            for m, d in job:
                machine_loads[m] += d
        
        lower_bound = max(max(job_durations), max(machine_loads) if machine_loads else 0)
        horizon = sum(job_durations)  # Upper bound
        
        model = cp_model.CpModel()
        
        all_starts = [[None] * len(job) for job in jobs_data]
        all_ends = [[None] * len(job) for job in jobs_data]
        machine_to_intervals = [[] for _ in range(M)]
        
        for j, job in enumerate(jobs_data):
            for k, (m, p) in enumerate(job):
                # Use tighter bounds: earliest start, latest end
                lb = early_start[j][k]
                ub = horizon - (job_durations[j] - early_start[j][k] - p)
                start = model.NewIntVar(lb, ub, f"s{j}_{k}")
                end = model.NewIntVar(lb + p, ub + p, f"e{j}_{k}")
                interval = model.NewIntervalVar(start, p, end, f"i{j}_{k}")
                all_starts[j][k] = start
                all_ends[j][k] = end
                machine_to_intervals[m].append(interval)
                if k > 0:
                    model.Add(start >= all_ends[j][k-1])
        
        for m in range(M):
            if machine_to_intervals[m]:
                model.AddNoOverlap(machine_to_intervals[m])
        
        makespan = model.NewIntVar(lower_bound, horizon, "makespan")
        model.AddMaxEquality(makespan, [all_ends[j][-1] for j in range(J)])
        model.Minimize(makespan)
        
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.linearization_level = 0
        solver.parameters.cp_model_presolve = True
        solver.parameters.cp_model_probing_level = 0
        
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [[int(solver.Value(all_starts[j][k])) for k in range(len(jobs_data[j]))] for j in range(J)]
        return []