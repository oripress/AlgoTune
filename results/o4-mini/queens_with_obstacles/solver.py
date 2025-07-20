import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: np.ndarray, **kwargs):
        """
        Solve the Queens with Obstacles problem using CP-SAT with segment constraints and hint.
        """
        instance = problem
        n, m = instance.shape
        model = cp_model.CpModel()
        # Create variables for each empty cell
        queens = {}
        for r in range(n):
            for c in range(m):
                if not instance[r, c]:
                    queens[(r, c)] = model.NewBoolVar(f"q_{r}_{c}")
        # Greedy initial solution hint
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
        attacked = instance.copy()
        hint_cells = []
        for r in range(n):
            for c in range(m):
                if not instance[r, c] and not attacked[r, c]:
                    hint_cells.append((r, c))
                    attacked[r, c] = True
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        while 0 <= nr < n and 0 <= nc < m and not instance[nr, nc]:
                            attacked[nr, nc] = True
                            nr += dr
                            nc += dc
        hint_set = set(hint_cells)
        for (r, c), var in queens.items():
            model.AddHint(var, int((r, c) in hint_set))
        # Row segments: at most one queen per unobstructed run
        for r in range(n):
            c0 = 0
            while c0 < m:
                while c0 < m and instance[r, c0]:
                    c0 += 1
                if c0 >= m:
                    break
                seg = []
                while c0 < m and not instance[r, c0]:
                    seg.append(queens[(r, c0)])
                    c0 += 1
                if len(seg) > 1:
                    model.AddAtMostOne(seg)
        # Column segments
        for c in range(m):
            r0 = 0
            while r0 < n:
                while r0 < n and instance[r0, c]:
                    r0 += 1
                if r0 >= n:
                    break
                seg = []
                while r0 < n and not instance[r0, c]:
                    seg.append(queens[(r0, c)])
                    r0 += 1
                if len(seg) > 1:
                    model.AddAtMostOne(seg)
        # Main diagonal segments (r - c constant)
        for d in range(-m + 1, n):
            r0 = max(d, 0)
            c0 = r0 - d
            seg = []
            while r0 < n and c0 < m:
                if instance[r0, c0]:
                    if len(seg) > 1:
                        model.AddAtMostOne(seg)
                    seg = []
                else:
                    seg.append(queens[(r0, c0)])
                r0 += 1
                c0 += 1
            if len(seg) > 1:
                model.AddAtMostOne(seg)
        # Anti-diagonal segments (r + c constant)
        for s in range(n + m - 1):
            r0 = max(0, s - (m - 1))
            c0 = s - r0
            seg = []
            while r0 < n and c0 >= 0:
                if instance[r0, c0]:
                    if len(seg) > 1:
                        model.AddAtMostOne(seg)
                    seg = []
                else:
                    seg.append(queens[(r0, c0)])
                r0 += 1
                c0 -= 1
            if len(seg) > 1:
                model.AddAtMostOne(seg)
        # Objective: maximize total queens
        model.Maximize(sum(queens.values()))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = kwargs.get("num_workers", 8)
        status = solver.Solve(model)
        # Extract solution
        result = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for (r, c), var in queens.items():
                if solver.Value(var):
                    result.append((r, c))
        return result