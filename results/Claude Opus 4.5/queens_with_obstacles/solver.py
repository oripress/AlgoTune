from typing import Any
import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: np.ndarray, **kwargs) -> Any:
        instance = problem
        n, m = instance.shape
        model = cp_model.CpModel()

        # Decision variables - only for valid cells, using integer index
        queens = {}
        valid_cells = []
        idx = 0
        for r in range(n):
            for c in range(m):
                if not instance[r, c]:
                    queens[(r, c)] = model.NewBoolVar(f"{idx}")
                    valid_cells.append((r, c))
                    idx += 1
        
        if not valid_cells:
            return []

        # Row constraints
        for r in range(n):
            seg = []
            for c in range(m):
                if instance[r, c]:
                    if len(seg) > 1:
                        model.AddAtMostOne(seg)
                    seg = []
                elif (r, c) in queens:
                    seg.append(queens[(r, c)])
            if len(seg) > 1:
                model.AddAtMostOne(seg)

        # Column constraints
        for c in range(m):
            seg = []
            for r in range(n):
                if instance[r, c]:
                    if len(seg) > 1:
                        model.AddAtMostOne(seg)
                    seg = []
                elif (r, c) in queens:
                    seg.append(queens[(r, c)])
            if len(seg) > 1:
                model.AddAtMostOne(seg)

        # Main diagonal constraints
        for k in range(-(n-1), m):
            seg = []
            for r in range(n):
                c = r - k
                if 0 <= c < m:
                    if instance[r, c]:
                        if len(seg) > 1:
                            model.AddAtMostOne(seg)
                        seg = []
                    elif (r, c) in queens:
                        seg.append(queens[(r, c)])
            if len(seg) > 1:
                model.AddAtMostOne(seg)

        # Anti-diagonal constraints
        for k in range(n + m - 1):
            seg = []
            for r in range(n):
                c = k - r
                if 0 <= c < m:
                    if instance[r, c]:
                        if len(seg) > 1:
                            model.AddAtMostOne(seg)
                        seg = []
                    elif (r, c) in queens:
                        seg.append(queens[(r, c)])
            if len(seg) > 1:
                model.AddAtMostOne(seg)

        # Maximize
        model.Maximize(sum(queens.values()))

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        solver.parameters.num_search_workers = 8
        solver.parameters.linearization_level = 0
        solver.parameters.cp_model_probing_level = 0
        solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(r, c) for (r, c), v in queens.items() if solver.Value(v)]
        else:
            return []