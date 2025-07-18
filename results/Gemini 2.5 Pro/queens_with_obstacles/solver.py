import numpy as np
from ortools.sat.python import cp_model
from typing import Any

class Solver:
    def solve(self, problem: np.ndarray, **kwargs) -> Any:
        """
        Solves the Queens with Obstacles Problem using an efficient CP-SAT model
        with parallel search.
        """
        instance = problem
        n, m = instance.shape
        model = cp_model.CpModel()

        # Decision variables for non-obstacle cells
        valid_cells = [(r, c) for r in range(n) for c in range(m) if not instance[r, c]]

        if not valid_cells:
            return []

        queens = {cell: model.NewBoolVar(f"q_{cell[0]}_{cell[1]}") for cell in valid_cells}

        # Helper to add constraints, avoiding repeated code.
        def add_at_most_one_constraint(segment):
            if len(segment) > 1:
                model.AddAtMostOne(segment)

        # Horizontal segments
        for r in range(n):
            segment = []
            for c in range(m):
                if not instance[r, c]:
                    segment.append(queens[(r, c)])
                else:
                    add_at_most_one_constraint(segment)
                    segment = []
            add_at_most_one_constraint(segment)

        # Vertical segments
        for c in range(m):
            segment = []
            for r in range(n):
                if not instance[r, c]:
                    segment.append(queens[(r, c)])
                else:
                    add_at_most_one_constraint(segment)
                    segment = []
            add_at_most_one_constraint(segment)

        # Diagonal segments (r-c = k)
        for k in range(1 - m, n):
            segment = []
            r_start, c_start = max(0, -k), max(0, k)
            r, c = r_start, c_start
            while r < n and c < m:
                if not instance[r, c]:
                    segment.append(queens[(r, c)])
                else:
                    add_at_most_one_constraint(segment)
                    segment = []
                r += 1
                c += 1
            add_at_most_one_constraint(segment)

        # Anti-diagonal segments (r+c = k)
        for k in range(n + m - 1):
            segment = []
            r_start = max(0, k - m + 1)
            c_start = k - r_start
            r, c = r_start, c_start
            while r < n and c >= 0:
                if not instance[r, c]:
                    segment.append(queens[(r, c)])
                else:
                    add_at_most_one_constraint(segment)
                    segment = []
                r += 1
                c -= 1
            add_at_most_one_constraint(segment)

        # Objective: Maximize the number of queens
        model.Maximize(sum(queens.values()))

        # Solve the model
        solver = cp_model.CpSolver()
        # Use a moderate number of workers for parallel search.
        # This is a balance between parallel speedup and overhead.
        solver.parameters.num_search_workers = 4
        status = solver.Solve(model)

        solution = []
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for cell in valid_cells:
                if solver.Value(queens[cell]):
                    solution.append(cell)
        
        return solution