from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the Queens with Obstacles Problem.

        Given an n x m boolean matrix where True denotes an obstacle, place the maximum
        number of queens so that no two queens attack each other (obstacles block attacks).

        This implementation uses a compact exact CP-SAT model with "at most one" constraints
        on contiguous obstacle-free segments along rows, columns, and both diagonals.
        """
        instance = np.asarray(problem, dtype=bool)
        n, m = instance.shape
        if n == 0 or m == 0:
            return []

        # Prepare segment id arrays; -1 indicates obstacle.
        row_seg = np.full((n, m), -1, dtype=np.int32)
        col_seg = np.full((n, m), -1, dtype=np.int32)
        diag_seg = np.full((n, m), -1, dtype=np.int32)   # r-c constant, slope +1
        adiag_seg = np.full((n, m), -1, dtype=np.int32)  # r+c constant, slope -1

        # Label row segments (contiguous empty cells between obstacles on each row)
        row_count = 0
        for r in range(n):
            current = -1
            open_seg = False
            for c in range(m):
                if instance[r, c]:
                    open_seg = False
                    continue
                if not open_seg:
                    current = row_count
                    row_count += 1
                    open_seg = True
                row_seg[r, c] = current

        # Label column segments
        col_count = 0
        for c in range(m):
            current = -1
            open_seg = False
            for r in range(n):
                if instance[r, c]:
                    open_seg = False
                    continue
                if not open_seg:
                    current = col_count
                    col_count += 1
                    open_seg = True
                col_seg[r, c] = current

        # Label diagonals with slope +1 (r-c = const)
        diag_count = 0
        for d in range(-(m - 1), n):
            r = d if d >= 0 else 0
            c = r - d
            current = -1
            open_seg = False
            while r < n and c < m:
                if instance[r, c]:
                    open_seg = False
                else:
                    if not open_seg:
                        current = diag_count
                        diag_count += 1
                        open_seg = True
                    diag_seg[r, c] = current
                r += 1
                c += 1

        # Label anti-diagonals with slope -1 (r+c = const)
        adiag_count = 0
        for s in range(n + m - 1):
            r = 0 if s < m else s - (m - 1)
            c = s - r
            current = -1
            open_seg = False
            while r < n and c >= 0:
                if c < m:
                    if instance[r, c]:
                        open_seg = False
                    else:
                        if not open_seg:
                            current = adiag_count
                            adiag_count += 1
                            open_seg = True
                        adiag_seg[r, c] = current
                # Move to next cell on anti-diagonal
                r += 1
                c -= 1

        # Build CP-SAT model
        model = cp_model.CpModel()

        # Create variables only for empty cells
        var_index = -np.ones((n, m), dtype=np.int32)
        vars_list: List[cp_model.IntVar] = []
        for r in range(n):
            for c in range(m):
                if not instance[r, c]:
                    idx = len(vars_list)
                    var_index[r, c] = idx
                    # Use compact unique names for performance
                    vars_list.append(model.NewBoolVar(f"x{idx}"))

        # If no variables (all obstacles), return empty solution
        if not vars_list:
            return []

        # Group variables by segments
        row_groups: List[List[int]] = [[] for _ in range(row_count)]
        col_groups: List[List[int]] = [[] for _ in range(col_count)]
        diag_groups: List[List[int]] = [[] for _ in range(diag_count)]
        adiag_groups: List[List[int]] = [[] for _ in range(adiag_count)]

        for r in range(n):
            for c in range(m):
                if instance[r, c]:
                    continue
                idx = var_index[r, c]
                rs = row_seg[r, c]
                cs = col_seg[r, c]
                ds = diag_seg[r, c]
                asg = adiag_seg[r, c]
                if rs >= 0:
                    row_groups[rs].append(idx)
                if cs >= 0:
                    col_groups[cs].append(idx)
                if ds >= 0:
                    diag_groups[ds].append(idx)
                if asg >= 0:
                    adiag_groups[asg].append(idx)

        # Add "at most one" constraints per segment (efficient in CP-SAT)
        def add_groups(groups: List[List[int]]):
            for group in groups:
                if len(group) > 1:
                    model.AddAtMostOne([vars_list[i] for i in group])

        add_groups(row_groups)
        add_groups(col_groups)
        add_groups(diag_groups)
        add_groups(adiag_groups)

        # Objective: maximize number of queens placed
        model.Maximize(sum(vars_list))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        solver.parameters.num_search_workers = 8  # Use parallelism if available

        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return []

        # Extract solution
        solution: List[Tuple[int, int]] = []
        for r in range(n):
            for c in range(m):
                idx = var_index[r, c]
                if idx >= 0 and solver.Value(vars_list[idx]) == 1:
                    solution.append((int(r), int(c)))

        return solution