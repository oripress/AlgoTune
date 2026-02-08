import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        instance = np.asarray(problem, dtype=bool)
        if instance.ndim == 0 or instance.size == 0:
            return []
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)
        n, m = instance.shape
        
        model = cp_model.CpModel()
        
        # Precompute obstacle grid for fast access
        obs = instance
        
        # Decision variables - only for non-obstacle cells
        queens = {}
        cells = []
        for r in range(n):
            for c in range(m):
                if not obs[r, c]:
                    v = model.NewBoolVar(f"q{r}_{c}")
                    queens[(r, c)] = v
                    cells.append((r, c))
        
        if not cells:
            return []
        
        def process_line(line_iter):
            seg = []
            for r, c in line_iter:
                if obs[r, c]:
                    if len(seg) > 1:
                        model.AddAtMostOne(seg)
                    seg = []
                else:
                    seg.append(queens[(r, c)])
            if len(seg) > 1:
                model.AddAtMostOne(seg)
        
        # Row segments
        for r in range(n):
            process_line(((r, c) for c in range(m)))
        
        # Column segments
        for c in range(m):
            process_line(((r, c) for r in range(n)))
        
        # Down-right diagonal segments
        for start in range(-(n-1), m):
            if start >= 0:
                sr, sc = 0, start
            else:
                sr, sc = -start, 0
            process_line(((sr+i, sc+i) for i in range(min(n-sr, m-sc))))
        
        # Anti-diagonal segments (down-left)
        for start in range(0, n + m - 1):
            if start < m:
                sr, sc = 0, start
            else:
                sr, sc = start - m + 1, m - 1
            process_line(((sr+i, sc-i) for i in range(min(n-sr, sc+1))))
        
        # Maximize the number of queens
        model.Maximize(sum(queens[rc] for rc in cells))
        
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        solver.parameters.num_workers = 8
        solver.parameters.cp_model_presolve = True
        solver.parameters.symmetry_level = 0
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(r, c) for r, c in cells if solver.Value(queens[(r, c)])]
        else:
            return []