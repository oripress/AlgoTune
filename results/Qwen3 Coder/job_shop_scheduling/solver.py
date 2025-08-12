import random
from typing import Any, List, Tuple, Dict
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[List[int]]:
        """
        Solve the JSSP using CP-SAT with interval variables and no-overlap.
        
        :param problem: Dict with "num_machines" and "jobs".
        :return: A list of J lists of start times for each operation.
        """
        M = problem["num_machines"]
        jobs_data = problem["jobs"]
        
        model = cp_model.CpModel()
        
        # Compute horizon as the sum of all durations
        horizon = sum(d for job in jobs_data for _, d in job)
        
        # Create interval vars and precedence constraints
        all_tasks = {}  # (j,k) -> (start_var, end_var, duration)
        machine_to_intervals: Dict[int, List[cp_model.IntervalVar]] = {m: [] for m in range(M)}
        
        for j, job in enumerate(jobs_data):
            for k, (m, p) in enumerate(job):
                suffix = f"_{j}_{k}"
                start = model.NewIntVar(0, horizon, f"start{suffix}")
                end = model.NewIntVar(0, horizon, f"end{suffix}")
                interval = model.NewIntervalVar(start, p, end, f"interval{suffix}")
                all_tasks[(j, k)] = (start, end, p)
                machine_to_intervals[m].append(interval)
                
                # Precedence constraint
                if k > 0:
                    prev_end = all_tasks[(j, k - 1)][1]
                    model.Add(start >= prev_end)
        
        # No-overlap on each machine
        for m in range(M):
            if len(machine_to_intervals[m]) > 1:  # Only add if there are multiple intervals
                model.AddNoOverlap(machine_to_intervals[m])
        
        # Makespan objective
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ends = []
        for job_id, job in enumerate(jobs_data):
            if job:  # Check if job is not empty
                _, end_var, _ = all_tasks[(job_id, len(job) - 1)]
                last_ends.append(end_var)
        if last_ends:
            model.AddMaxEquality(makespan, last_ends)
            model.Minimize(makespan)
        
        # Solve with optimized parameters for speed
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 0.5
        solver.parameters.num_workers = 1
        solver.parameters.log_search_progress = False
        solver.parameters.cp_model_probing_level = 0
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solution = []
            for j, job in enumerate(jobs_data):
                starts = []
                for k, _ in enumerate(job):
                    starts.append(int(solver.Value(all_tasks[(j, k)][0])))
                solution.append(starts)
            return solution
        else:
            # Return a trivial solution if no solution found
            return [[] for _ in jobs_data]