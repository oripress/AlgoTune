import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        n, m = problem.shape
        model = cp_model.CpModel()
        
        # Create grid of variables
        queens = [[model.NewBoolVar(f'q_{r}_{c}') for c in range(m)] for r in range(n)]
        
        # Set obstacles to 0
        for r in range(n):
            for c in range(m):
                if problem[r, c]:
                    model.Add(queens[r][c] == 0)

        segments = []
        
        # Row segments
        for r in range(n):
            start = 0
            while start < m:
                # Skip obstacles
                while start < m and problem[r, start]:
                    start += 1
                end = start
                while end < m and not problem[r, end]:
                    end += 1
                if start < end:
                    seg = [(r, c) for c in range(start, end)]
                    segments.append(seg)
                start = end
        
        # Column segments
        for c in range(m):
            start = 0
            while start < n:
                # Skip obstacles
                while start < n and problem[start, c]:
                    start += 1
                end = start
                while end < n and not problem[end, c]:
                    end += 1
                if start < end:
                    seg = [(r, c) for r in range(start, end)]
                    segments.append(seg)
                start = end
        
        # Diagonal segments (top-left to bottom-right)
        for diag_sum in range(n + m - 1):
            r = max(0, diag_sum - m + 1)
            c = min(diag_sum, m - 1)
            while r < n and c >= 0:
                if not problem[r, c]:
                    seg = []
                    while r < n and c >= 0 and not problem[r, c]:
                        seg.append((r, c))
                        r += 1
                        c -= 1
                    if len(seg) > 1:
                        segments.append(seg)
                else:
                    r += 1
                    c -= 1
        
        # Diagonal segments (top-right to bottom-left)
        for diag_diff in range(-m + 1, n):
            r = max(0, diag_diff)
            c = max(0, -diag_diff)
            while r < n and c < m:
                if not problem[r, c]:
                    seg = []
                    while r < n and c < m and not problem[r, c]:
                        seg.append((r, c))
                        r += 1
                        c += 1
                    if len(seg) > 1:
                        segments.append(seg)
                else:
                    r += 1
                    c += 1
        
        # Add constraints: at most one queen per segment
        for seg in segments:
            if len(seg) > 1:
                model.AddAtMostOne(queens[r][c] for (r, c) in seg)

        # Maximize total queens
        total_queens = sum(queens[r][c] for r in range(n) for c in range(m))
        model.Maximize(total_queens)
        
        # Configure solver for maximum performance
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 4  # Parallel processing
        solver.parameters.log_search_progress = False
        
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(r, c) for r in range(n) for c in range(m) if solver.Value(queens[r][c]) == 1]
        else:
            return []