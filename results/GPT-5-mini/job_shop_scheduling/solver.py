from typing import Any, Dict, List, Tuple

# Try to import OR-Tools; fall back to None if unavailable.
try:
    from ortools.sat.python import cp_model  # type: ignore
except Exception:
    cp_model = None  # type: ignore

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[List[int]]:
        """
        Solve the Job Shop Scheduling Problem (JSSP).

        Uses OR-Tools CP-SAT (interval variables + no-overlap) when available to
        obtain an optimal schedule. If OR-Tools is not present, uses a
        deterministic greedy list-scheduling heuristic to produce a feasible
        schedule.

        Input:
          problem: {"num_machines": M, "jobs": [[(m,p), ...], ...]}

        Output:
          List of J lists of start times for each operation.
        """
        M = int(problem.get("num_machines", 0))
        jobs: List[List[Tuple[int, int]]] = problem.get("jobs", [])

        J = len(jobs)
        # Quick returns for degenerate cases
        if J == 0:
            return []
        if M <= 0:
            return [[] for _ in range(J)]

        # Compute horizon as total processing time (upper bound)
        horizon = sum(int(p) for job in jobs for (_m, p) in job)

        # If OR-Tools not available, use a deterministic greedy heuristic
        if cp_model is None:
            machine_ready = [0] * M
            job_ready = [0] * J
            next_op = [0] * J
            starts: List[List[int]] = [[] for _ in range(J)]
            remaining = sum(len(job) for job in jobs)

            # At each step, pick the next operation with minimal (est, -dur, job)
            while remaining > 0:
                best_key = None
                best_j = -1
                best_m = None
                best_p = None
                for j in range(J):
                    k = next_op[j]
                    if k >= len(jobs[j]):
                        continue
                    m, p = jobs[j][k]
                    est = max(job_ready[j], machine_ready[m])
                    key = (est, -int(p), j)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_j = j
                        best_m = m
                        best_p = int(p)
                if best_j < 0:
                    break
                s = max(job_ready[best_j], machine_ready[best_m])
                starts[best_j].append(int(s))
                e = int(s) + int(best_p)
                job_ready[best_j] = e
                machine_ready[best_m] = e
                next_op[best_j] += 1
                remaining -= 1
            return starts

        # Build CP-SAT model with interval variables
        model = cp_model.CpModel()
        all_tasks: Dict[Tuple[int, int], Tuple[Any, Any, int]] = {}
        machine_to_intervals: Dict[int, List[Any]] = {m: [] for m in range(M)}

        for j, job in enumerate(jobs):
            for k, (m, p) in enumerate(job):
                p_int = int(p)
                start = model.NewIntVar(0, horizon, f"start_{j}_{k}")
                end = model.NewIntVar(0, horizon, f"end_{j}_{k}")
                interval = model.NewIntervalVar(start, p_int, end, f"interval_{j}_{k}")
                all_tasks[(j, k)] = (start, end, p_int)
                # Add interval to machine list (assume machines indexed properly)
                if 0 <= m < M:
                    machine_to_intervals.setdefault(m, []).append(interval)
                else:
                    # Defensive: if machine index out of range, create slot dynamically
                    machine_to_intervals.setdefault(m, []).append(interval)
                # Precedence within job
                if k > 0:
                    prev_end = all_tasks[(j, k - 1)][1]
                    model.Add(start >= prev_end)

        # No-overlap on each machine
        for m, intervals in machine_to_intervals.items():
            if intervals:
                model.AddNoOverlap(intervals)

        # Makespan objective
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ends = []
        for j, job in enumerate(jobs):
            if len(job) > 0:
                last_ends.append(all_tasks[(j, len(job) - 1)][1])
        if last_ends:
            model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)

        solver = cp_model.CpSolver()

        # Allow optional time limit parameter only if explicitly provided.
        if "time_limit" in kwargs:
            try:
                t = float(kwargs["time_limit"])
                if t > 0:
                    solver.parameters.max_time_in_seconds = t
            except Exception:
                pass

        # Use multiple workers where possible (bounded to avoid oversubscription)
        try:
            import multiprocessing as _mp  # type: ignore
            solver.parameters.num_search_workers = min(8, max(1, _mp.cpu_count()))
        except Exception:
            pass

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            solution: List[List[int]] = []
            for j, job in enumerate(jobs):
                starts: List[int] = []
                for k, _ in enumerate(job):
                    starts.append(int(solver.Value(all_tasks[(j, k)][0])))
                solution.append(starts)
            return solution
        else:
            # Match reference behavior: return empty list if optimality not proven
            return []