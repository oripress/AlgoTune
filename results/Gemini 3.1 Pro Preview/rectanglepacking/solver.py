from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        if hasattr(problem, 'container_width'):
            W = problem.container_width
            H = problem.container_height
            rectangles = [(r.width, r.height, r.rotatable) for r in problem.rectangles]
        else:
            W, H, rectangles = problem[0], problem[1], problem[2]
            
        model = cp_model.CpModel()
        
        x_intervals = []
        y_intervals = []
        
        placed_vars = []
        rect_info = []
        
        for i, (w, h, r) in enumerate(rectangles):
            can_fit_nr = w <= W and h <= H
            can_fit_r = r and h <= W and w <= H and w != h
            
            vars_for_rect = []
            
            if can_fit_nr:
                placed_nr = model.NewBoolVar(f'placed_nr_{i}')
                x_start_nr = model.NewIntVar(0, W - w, f'x_start_nr_{i}')
                x_end_nr = model.NewIntVar(w, W, f'x_end_nr_{i}')
                x_int_nr = model.NewOptionalIntervalVar(x_start_nr, w, x_end_nr, placed_nr, f'x_int_nr_{i}')
                
                y_start_nr = model.NewIntVar(0, H - h, f'y_start_nr_{i}')
                y_end_nr = model.NewIntVar(h, H, f'y_end_nr_{i}')
                y_int_nr = model.NewOptionalIntervalVar(y_start_nr, h, y_end_nr, placed_nr, f'y_int_nr_{i}')
                
                x_intervals.append(x_int_nr)
                y_intervals.append(y_int_nr)
                vars_for_rect.append((placed_nr, x_start_nr, y_start_nr, False))
                
            if can_fit_r:
                placed_r = model.NewBoolVar(f'placed_r_{i}')
                x_start_r = model.NewIntVar(0, W - h, f'x_start_r_{i}')
                x_end_r = model.NewIntVar(h, W, f'x_end_r_{i}')
                x_int_r = model.NewOptionalIntervalVar(x_start_r, h, x_end_r, placed_r, f'x_int_r_{i}')
                
                y_start_r = model.NewIntVar(0, H - w, f'y_start_r_{i}')
                y_end_r = model.NewIntVar(w, H, f'y_end_r_{i}')
                y_int_r = model.NewOptionalIntervalVar(y_start_r, w, y_end_r, placed_r, f'y_int_r_{i}')
                
                x_intervals.append(x_int_r)
                y_intervals.append(y_int_r)
                vars_for_rect.append((placed_r, x_start_r, y_start_r, True))
                
            if len(vars_for_rect) > 1:
                model.AddAtMostOne([v[0] for v in vars_for_rect])
                
            if vars_for_rect:
                placed_vars.append([v[0] for v in vars_for_rect])
                rect_info.append((i, vars_for_rect))
                
        if x_intervals:
            model.AddNoOverlap2D(x_intervals, y_intervals)
        
        model.Maximize(sum(v for vars_list in placed_vars for v in vars_list))
        
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.linearization_level = 0
        solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
        solver.parameters.max_time_in_seconds = 900.0
        
        status = solver.Solve(model)
        
        solution = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i, vars_for_rect in rect_info:
                for placed_var, x_start, y_start, is_rotated in vars_for_rect:
                    if solver.Value(placed_var):
                        solution.append((i, solver.Value(x_start), solver.Value(y_start), is_rotated))
                        break
                        
        return solution