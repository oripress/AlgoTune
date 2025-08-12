import numpy as np
from ortools.sat.python import cp_model
import heapq

class Solver:
    def solve(self, problem, **kwargs):
        # Convert input to Python bool grid for faster access
        grid = [list(map(bool, row)) for row in problem]
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        # Build segment IDs for 4 directions
        # horizontal segments
        h_seg = [[-1] * m for _ in range(n)]
        seg = -1
        for r in range(n):
            blocked = True
            for c in range(m):
                if not grid[r][c]:
                    if blocked:
                        seg += 1
                        blocked = False
                    h_seg[r][c] = seg
                else:
                    blocked = True
        n_h = seg + 1
        # vertical segments
        v_seg = [[-1] * m for _ in range(n)]
        seg = -1
        for c in range(m):
            blocked = True
            for r in range(n):
                if not grid[r][c]:
                    if blocked:
                        seg += 1
                        blocked = False
                    v_seg[r][c] = seg
                else:
                    blocked = True
        n_v = seg + 1
        # main diagonal (\) segments
        d1_seg = [[-1] * m for _ in range(n)]
        seg = -1
        for d in range(-(m - 1), n):
            blocked = True
            if d >= 0:
                r0, c0 = d, 0
            else:
                r0, c0 = 0, -d
            r, c = r0, c0
            while r < n and c < m:
                if not grid[r][c]:
                    if blocked:
                        seg += 1
                        blocked = False
                    d1_seg[r][c] = seg
                else:
                    blocked = True
                r += 1
                c += 1
        n_d1 = seg + 1
        # anti-diagonal (/) segments
        d2_seg = [[-1] * m for _ in range(n)]
        seg = -1
        for s in range(n + m - 1):
            blocked = True
            if s < m:
                r0, c0 = 0, s
            else:
                r0, c0 = s - (m - 1), m - 1
            r, c = r0, c0
            while r < n and c >= 0:
                if not grid[r][c]:
                    if blocked:
                        seg += 1
                        blocked = False
                    d2_seg[r][c] = seg
                else:
                    blocked = True
                r += 1
                c -= 1
        n_d2 = seg + 1
        # Collect free cells and segment groups
        free_cells = []
        groups = {
            'h': [[] for _ in range(n_h)],
            'v': [[] for _ in range(n_v)],
            'd1': [[] for _ in range(n_d1)],
            'd2': [[] for _ in range(n_d2)]
        }
        for r in range(n):
            for c in range(m):
                if not grid[r][c]:
                    idx = len(free_cells)
                    free_cells.append((r, c))
                    groups['h'][h_seg[r][c]].append(idx)
                    groups['v'][v_seg[r][c]].append(idx)
                    groups['d1'][d1_seg[r][c]].append(idx)
                    groups['d2'][d2_seg[r][c]].append(idx)
        N = len(free_cells)
        # Build CP-SAT model
        model = cp_model.CpModel()
        queens = [model.NewBoolVar(f"q{i}") for i in range(N)]
        # Add segment constraints: at most one queen per segment
        for seg_list in groups.values():
            for ids in seg_list:
                if len(ids) > 1:
                    model.Add(sum(queens[i] for i in ids) <= 1)
        # Maximize total queens
        model.Maximize(sum(queens))
        # Solver parameters
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = kwargs.get("num_workers", 8)
        if "time_limit" in kwargs:
            solver.parameters.max_time_in_seconds = kwargs["time_limit"]
        # Solve
        status = solver.Solve(model)
        # Extract solution
        result = []
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for i, qc in enumerate(queens):
                if solver.Value(qc):
                    result.append(free_cells[i])
        return result