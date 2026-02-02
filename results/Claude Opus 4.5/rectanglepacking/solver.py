from ortools.sat.python import cp_model
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        W, H, rectangles = problem
        n = len(rectangles)
        
        if n == 0:
            return []
        
        model = cp_model.CpModel()
        
        placed_vars = [None] * n
        rotated_vars = [None] * n
        x_vars = [None] * n
        y_vars = [None] * n
        x_intervals = []
        y_intervals = []
        valid_indices = []
        
        for i, (w, h, can_rotate) in enumerate(rectangles):
            fits_normal = w <= W and h <= H
            fits_rotated = can_rotate and h <= W and w <= H
            
            if not fits_normal and not fits_rotated:
                continue
            
            valid_indices.append(i)
            p = model.new_bool_var(f"p_{i}")
            placed_vars[i] = p
            
            need_rotation = fits_normal and fits_rotated and w != h
            
            if need_rotation:
                r = model.new_bool_var(f"r_{i}")
                rotated_vars[i] = r
                
                x = model.new_int_var(0, W - min(w, h), f"x_{i}")
                y = model.new_int_var(0, H - min(w, h), f"y_{i}")
                
                sx = model.new_int_var(min(w, h), max(w, h), f"sx_{i}")
                sy = model.new_int_var(min(w, h), max(w, h), f"sy_{i}")
                ex = model.new_int_var(min(w, h), W, f"ex_{i}")
                ey = model.new_int_var(min(w, h), H, f"ey_{i}")
                
                model.add(sx == h).only_enforce_if(r)
                model.add(sy == w).only_enforce_if(r)
                model.add(sx == w).only_enforce_if(r.Not())
                model.add(sy == h).only_enforce_if(r.Not())
                
                model.add(ex == x + sx)
                model.add(ey == y + sy)
                model.add(ex <= W)
                model.add(ey <= H)
                
                x_int = model.new_optional_interval_var(x, sx, ex, p, f"xi_{i}")
                y_int = model.new_optional_interval_var(y, sy, ey, p, f"yi_{i}")
            else:
                if fits_normal:
                    rotated_vars[i] = False
                    eff_w, eff_h = w, h
                else:
                    rotated_vars[i] = True
                    eff_w, eff_h = h, w
                
                x = model.new_int_var(0, W - eff_w, f"x_{i}")
                y = model.new_int_var(0, H - eff_h, f"y_{i}")
                
                x_int = model.new_optional_fixed_size_interval_var(x, eff_w, p, f"xi_{i}")
                y_int = model.new_optional_fixed_size_interval_var(y, eff_h, p, f"yi_{i}")
            
            x_vars[i] = x
            y_vars[i] = y
            x_intervals.append(x_int)
            y_intervals.append(y_int)
        
        if not x_intervals:
            return []
        
        model.add_no_overlap_2d(x_intervals, y_intervals)
        model.maximize(sum(placed_vars[i] for i in valid_indices))
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30
        solver.parameters.num_workers = 8
        status = solver.solve(model)
        
        result = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i in valid_indices:
                if solver.value(placed_vars[i]):
                    r_val = rotated_vars[i] if isinstance(rotated_vars[i], bool) else solver.value(rotated_vars[i]) == 1
                    result.append((i, solver.value(x_vars[i]), solver.value(y_vars[i]), r_val))
        
        return result