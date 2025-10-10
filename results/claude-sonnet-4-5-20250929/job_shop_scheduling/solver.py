from typing import Any
from ortools.sat.python import cp_model
import numpy as np
from numba import njit

@njit(cache=True)
def solve_small_jssp(jobs_np, num_machines, num_jobs):
    """Fast backtracking solver for small JSSP instances."""
    # Count total operations
    total_ops = 0
    for j in range(num_jobs):
        for k in range(len(jobs_np[j])):
            if jobs_np[j][k][0] >= 0:  # Valid operation
                total_ops += 1
            else:
                break
    
    # Calculate horizon
    horizon = 0
    for j in range(num_jobs):
        job_dur = 0
        for k in range(len(jobs_np[j])):
            if jobs_np[j][k][0] >= 0:
                job_dur += jobs_np[j][k][1]
            else:
                break
        horizon = max(horizon, job_dur)
    
    # Greedy scheduling
    machine_free = np.zeros(num_machines, dtype=np.int32)
    job_next_op = np.zeros(num_jobs, dtype=np.int32)
    job_time = np.zeros(num_jobs, dtype=np.int32)
    starts = np.full((num_jobs, 20), -1, dtype=np.int32)  # Max 20 ops per job
    
    for _ in range(total_ops):
        # Find operation with earliest possible start
        best_j = -1
        best_start = horizon * 10
        
        for j in range(num_jobs):
            k = job_next_op[j]
            if k < len(jobs_np[j]) and jobs_np[j][k][0] >= 0:
                m = jobs_np[j][k][0]
                p = jobs_np[j][k][1]
                start = max(job_time[j], machine_free[m])
                if start < best_start:
                    best_start = start
                    best_j = j
        
        if best_j >= 0:
            j = best_j
            k = job_next_op[j]
            m = jobs_np[j][k][0]
            p = jobs_np[j][k][1]
            start = max(job_time[j], machine_free[m])
            
            starts[j][k] = start
            job_time[j] = start + p
            machine_free[m] = start + p
            job_next_op[j] += 1
    
    return starts

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        """
        Solve the JSSP with hybrid approach.
        """
        M = problem["num_machines"]
        jobs_data = problem["jobs"]
        num_jobs = len(jobs_data)
        
        # For small problems, use fast numba solver
        if num_jobs <= 10 and M <= 10:
            # Convert to numpy for numba
            max_ops = max(len(job) for job in jobs_data)
            jobs_np = np.full((num_jobs, max_ops, 2), -1, dtype=np.int32)
            for j, job in enumerate(jobs_data):
                for k, (m, p) in enumerate(job):
                    jobs_np[j, k, 0] = m
                    jobs_np[j, k, 1] = p
            
            starts_np = solve_small_jssp(jobs_np, M, num_jobs)
            
            # Convert back to list
            solution = []
            for j in range(num_jobs):
                job_starts = []
                for k in range(len(jobs_data[j])):
                    job_starts.append(int(starts_np[j, k]))
                solution.append(job_starts)
            return solution
        
        # For larger problems, use CP-SAT
        machine_free = [0] * M
        job_time = [0] * len(jobs_data)
        
        for j, job in enumerate(jobs_data):
            for k, (m, p) in enumerate(job):
                start = max(job_time[j], machine_free[m])
                job_time[j] = start + p
                machine_free[m] = start + p
        
        horizon = max(job_time)
        model = cp_model.CpModel()
        
        earliest = {}
        for j, job in enumerate(jobs_data):
            t = 0
            for k, (m, p) in enumerate(job):
                earliest[(j, k)] = t
                t += p
        
        all_tasks = {}
        machine_to_intervals = {m: [] for m in range(M)}
        
        for j, job in enumerate(jobs_data):
            for k, (m, p) in enumerate(job):
                lb = earliest[(j, k)]
                start = model.NewIntVar(lb, horizon, "")
                end = model.NewIntVar(lb + p, horizon + p, "")
                interval = model.NewIntervalVar(start, p, end, "")
                all_tasks[(j, k)] = (start, end, p)
                machine_to_intervals[m].append(interval)
                if k > 0:
                    prev_end = all_tasks[(j, k - 1)][1]
                    model.Add(start >= prev_end)
        
        for m in range(M):
            model.AddNoOverlap(machine_to_intervals[m])
        
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ends = [all_tasks[(j, len(job) - 1)][1] for j, job in enumerate(jobs_data)]
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)
        
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.log_search_progress = False
        solver.parameters.cp_model_presolve = True
        solver.parameters.linearization_level = 0
        solver.parameters.cp_model_probing_level = 0
        
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solution = []
            for j, job in enumerate(jobs_data):
                starts = [int(solver.Value(all_tasks[(j, k)][0])) for k in range(len(job))]
                solution.append(starts)
            return solution
        else:
            return []