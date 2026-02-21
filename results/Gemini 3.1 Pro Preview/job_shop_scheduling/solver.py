from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[list[int]]:
        M = problem["num_machines"]
        jobs_data = problem["jobs"]

        model = cp_model.CpModel()
        horizon = sum(d for job in jobs_data for _, d in job)

        all_tasks = []
        machine_to_intervals = [[] for _ in range(M)]
        
        for j, job in enumerate(jobs_data):
            job_tasks = []
            prev_start = None
            prev_p = 0
            for m, p in job:
                start = model.NewIntVar(0, horizon, "")
                interval = model.NewFixedSizeIntervalVar(start, p, "")
                job_tasks.append(start)
                machine_to_intervals[m].append(interval)
                if prev_start is not None:
                    model.Add(start - prev_start >= prev_p)
                prev_start = start
                prev_p = p
            all_tasks.append(job_tasks)

        for m in range(M):
            model.AddNoOverlap(machine_to_intervals[m])

        makespan = model.NewIntVar(0, horizon, "")
        last_ends = []
        for job_id, job in enumerate(jobs_data):
            last_ends.append(all_tasks[job_id][-1] + job[-1][1])
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.cp_model_presolve = False
        solver.parameters.linearization_level = 0
        solver.parameters.catch_sigint_signal = False
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            solution = []
            for j, job in enumerate(jobs_data):
                starts = []
                for k, _ in enumerate(job):
                    starts.append(int(solver.Value(all_tasks[j][k])))
                solution.append(starts)
            return solution
        else:
            return []