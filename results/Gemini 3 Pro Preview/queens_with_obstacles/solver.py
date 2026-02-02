from collections.abc import Iterator
import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem) -> list[tuple[int, int]]:
        if not hasattr(problem, 'shape'):
            instance = np.array(problem)
        else:
            instance = problem
        n, m = instance.shape
        model = cp_model.CpModel()

        # Variables
        queens = {}
        for r in range(n):
            for c in range(m):
                if not instance[r, c]:
                    queens[(r, c)] = model.NewBoolVar(f"q_{r}_{c}")

        # Helper to process a sequence of coordinates
        def add_segment_constraints(sequence):
            segment = []
            for r, c in sequence:
                if instance[r, c]: # Obstacle
                    if len(segment) > 1:
                        model.Add(sum(queens[pos] for pos in segment) <= 1)
                    segment = []
                else:
                    segment.append((r, c))
            if len(segment) > 1:
                model.Add(sum(queens[pos] for pos in segment) <= 1)

        # Rows
        for r in range(n):
            add_segment_constraints(((r, c) for c in range(m)))

        # Columns
        for c in range(m):
            add_segment_constraints(((r, c) for r in range(n)))

        # Diagonals (down-right)
        # r - c = k. k in [-(m-1), n-1]
        for k in range(-(m - 1), n):
            r_start = max(0, k)
            r_end = min(n, m + k)
            if r_start < r_end:
                add_segment_constraints(((r, r - k) for r in range(r_start, r_end)))

        # Anti-diagonals (down-left)
        # r + c = k. k in [0, n + m - 2]
        for k in range(n + m - 1):
            r_start = max(0, k - m + 1)
            r_end = min(n, k + 1)
            if r_start < r_end:
                add_segment_constraints(((r, k - r) for r in range(r_start, r_end)))

        # Objective
        model.Maximize(sum(queens.values()))

        solver = cp_model.CpSolver()
        # Use multiple workers for speedup
        solver.parameters.num_search_workers = 8
        
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [pos for pos, var in queens.items() if solver.Value(var)]
        return []