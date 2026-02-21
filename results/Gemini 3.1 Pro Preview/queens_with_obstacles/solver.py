import numpy as np
from ortools.sat.python import cp_model
from typing import Any

class Solver:
    def solve(self, problem: np.ndarray, **kwargs) -> Any:
        n, m = problem.shape
        model = cp_model.CpModel()
        
        empty = ~problem
        queens = [[None] * m for _ in range(n)]
        vars_list = []
        
        for r in range(n):
            for c in range(m):
                if empty[r, c]:
                    v = model.NewBoolVar("")
                    queens[r][c] = v
                    vars_list.append(v)
                    
        if not vars_list:
            return []
            
        # 1. Horizontal segments
        for r in range(n):
            segment = []
            for c in range(m):
                if empty[r, c]:
                    segment.append(queens[r][c])
                else:
                    if len(segment) == 2:
                        model.AddBoolOr([segment[0].Not(), segment[1].Not()])
                    elif len(segment) > 2:
                        model.AddAtMostOne(segment)
                    segment = []
            if len(segment) == 2:
                model.AddBoolOr([segment[0].Not(), segment[1].Not()])
            elif len(segment) > 2:
                model.AddAtMostOne(segment)
                
        # 2. Vertical segments
        for c in range(m):
            segment = []
            for r in range(n):
                if empty[r, c]:
                    segment.append(queens[r][c])
                else:
                    if len(segment) == 2:
                        model.AddBoolOr([segment[0].Not(), segment[1].Not()])
                    elif len(segment) > 2:
                        model.AddAtMostOne(segment)
                    segment = []
            if len(segment) == 2:
                model.AddBoolOr([segment[0].Not(), segment[1].Not()])
            elif len(segment) > 2:
                model.AddAtMostOne(segment)
                
        # 3. Diagonal 1 (r - c = d)
        for d in range(-(m - 1), n):
            segment = []
            start_r = max(0, d)
            end_r = min(n, m + d)
            for r in range(start_r, end_r):
                c = r - d
                if empty[r, c]:
                    segment.append(queens[r][c])
                else:
                    if len(segment) == 2:
                        model.AddBoolOr([segment[0].Not(), segment[1].Not()])
                    elif len(segment) > 2:
                        model.AddAtMostOne(segment)
                    segment = []
            if len(segment) == 2:
                model.AddBoolOr([segment[0].Not(), segment[1].Not()])
            elif len(segment) > 2:
                model.AddAtMostOne(segment)
                
        # 4. Diagonal 2 (r + c = d)
        for d in range(n + m - 1):
            segment = []
            start_r = max(0, d - m + 1)
            end_r = min(n, d + 1)
            for r in range(start_r, end_r):
                c = d - r
                if empty[r, c]:
                    segment.append(queens[r][c])
                else:
                    if len(segment) == 2:
                        model.AddBoolOr([segment[0].Not(), segment[1].Not()])
                    elif len(segment) > 2:
                        model.AddAtMostOne(segment)
                    segment = []
            if len(segment) == 2:
                model.AddBoolOr([segment[0].Not(), segment[1].Not()])
            elif len(segment) > 2:
                model.AddAtMostOne(segment)
                
        model.Maximize(sum(vars_list))
        
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 4
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            res = []
            for r in range(n):
                for c in range(m):
                    if empty[r, c] and solver.Value(queens[r][c]):
                        res.append((r, c))
            return res
        return []