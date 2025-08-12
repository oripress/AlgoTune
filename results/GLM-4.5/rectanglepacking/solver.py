from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        W, H, rectangles = problem
        n = len(rectangles)
        model = cp_model.CpModel()
        
        # Create variables for each rectangle
        placed_vars = [model.new_bool_var(f'placed_{i}') for i in range(n)]
        rotated_vars = [model.new_bool_var(f'rotated_{i}') for i in range(n)]
        
        # Position and size variables
        x_starts = []
        x_ends = []
        y_starts = []
        y_ends = []
        x_sizes = []
        y_sizes = []
        x_intervals = []
        y_intervals = []
        
        for i, (w, h, rotatable) in enumerate(rectangles):
            # Position variables
            x_start = model.new_int_var(0, W, f'x_start_{i}')
            x_end = model.new_int_var(0, W, f'x_end_{i}')
            y_start = model.new_int_var(0, H, f'y_start_{i}')
            y_end = model.new_int_var(0, H, f'y_end_{i}')
            x_starts.append(x_start)
            x_ends.append(x_end)
            y_starts.append(y_start)
            y_ends.append(y_end)
            
            # Size variables
            if rotatable:
                x_size = model.new_int_var(min(w, h), max(w, h), f'x_size_{i}')
                y_size = model.new_int_var(min(w, h), max(w, h), f'y_size_{i}')
                
                # Link sizes to rotation
                model.add(x_size == w).only_enforce_if(rotated_vars[i].Not())
                model.add(x_size == h).only_enforce_if(rotated_vars[i])
                model.add(y_size == h).only_enforce_if(rotated_vars[i].Not())
                model.add(y_size == w).only_enforce_if(rotated_vars[i])
            else:
                x_size = model.new_int_var(w, w, f'x_size_{i}')
                y_size = model.new_int_var(h, h, f'y_size_{i}')
                model.add(rotated_vars[i] == 0)
            
            x_sizes.append(x_size)
            y_sizes.append(y_size)
            
            # Link positions to sizes
            model.add(x_end == x_start + x_size).only_enforce_if(placed_vars[i])
            model.add(y_end == y_start + y_size).only_enforce_if(placed_vars[i])
            
            # If not placed, set all to 0
            model.add(x_start == 0).only_enforce_if(placed_vars[i].Not())
            model.add(y_start == 0).only_enforce_if(placed_vars[i].Not())
            model.add(x_end == 0).only_enforce_if(placed_vars[i].Not())
            model.add(y_end == 0).only_enforce_if(placed_vars[i].Not())
            
            # Create interval variables
            x_interval = model.new_optional_interval_var(x_start, x_size, x_end, placed_vars[i], f'x_interval_{i}')
            y_interval = model.new_optional_interval_var(y_start, y_size, y_end, placed_vars[i], f'y_interval_{i}')
            x_intervals.append(x_interval)
            y_intervals.append(y_interval)
        
        # Add 2D no-overlap constraint - this is much more efficient than pairwise constraints
        model.add_no_overlap_2d(x_intervals, y_intervals)
        
        # Container bounds
        for i in range(n):
            model.add(x_ends[i] <= W).only_enforce_if(placed_vars[i])
            model.add(y_ends[i] <= H).only_enforce_if(placed_vars[i])
        
        # Objective: maximize number of placed rectangles
        model.maximize(sum(placed_vars))
        
        # Solve with optimized parameters
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 900.0
        solver.parameters.num_search_workers = 8
        solver.parameters.log_search_progress = False
        
        status = solver.solve(model)
        
        # Extract solution
        solution = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i in range(n):
                if solver.value(placed_vars[i]):
                    x = solver.value(x_starts[i])
                    y = solver.value(y_starts[i])
                    rotated = solver.value(rotated_vars[i]) == 1
                    solution.append((i, x, y, rotated))
        
        return solution