from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[list[int]]:
        """
        Solve the JSSP using CP-SAT with optimizations.
        
        :param problem: Dict with "num_machines" and "jobs".
        :return: A list of J lists of start times for each operation.
        """
        M = problem["num_machines"]
        jobs_data = problem["jobs"]
        
        model = cp_model.CpModel()
        
        # Compute horizon
        horizon = sum(d for job in jobs_data for _, d in job)
        
        # Create interval vars and precedence constraints
        all_tasks = {}  # (j,k) -> (start_var, end_var, duration)
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
        
        # No-overlap on each machine
        for m in range(M):
            model.AddNoOverlap(machine_to_intervals[m])
        
        # Makespan objective
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ends = []
        for job_id, job in enumerate(jobs_data):
            _, end_var, _ = all_tasks[(job_id, len(job) - 1)]
            last_ends.append(end_var)
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)
        
        # Solver with optimizations
        solver = cp_model.CpSolver()
        
        # Set search parameters for faster solving
        solver.parameters.num_search_workers = 8  # Use multiple threads
        solver.parameters.linearization_level = 2  # More aggressive linearization
        solver.parameters.cp_model_presolve = True
        solver.parameters.max_time_in_seconds = 10.0  # Reasonable time limit
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = []
            for j, job in enumerate(jobs_data):
                starts = []
                for k, _ in enumerate(job):
                    starts.append(int(solver.Value(all_tasks[(j, k)][0])))
                solution.append(starts)
            return solution
        else:
            return []