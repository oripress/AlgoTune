from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        W, H, rectangles = problem
        n = len(rectangles)
        
        # Sort rectangles by area (largest first) to improve solver performance
        sorted_indices = sorted(range(n), key=lambda i: rectangles[i][0] * rectangles[i][1], reverse=True)
        sorted_rectangles = [rectangles[i] for i in sorted_indices]
        
        model = cp_model.CpModel()
        
        # Create decision variables
        placed = [model.NewBoolVar(f"placed_{i}") for i in range(n)]
        rotated = [model.NewBoolVar(f"rotated_{i}") for i in range(n)]
        x = [model.NewIntVar(0, W, f"x_{i}") for i in range(n)]
        y = [model.NewIntVar(0, H, f"y_{i}") for i in range(n)]
        
        # Create interval variables for efficient 2D no-overlap
        x_intervals = []
        y_intervals = []
        
        for i, rect in enumerate(sorted_rectangles):
            w_val, h_val, r_flag = rect
            
            # For non-rotatable rectangles, fix rotation to False
            if not r_flag:
                model.Add(rotated[i] == 0)
            
            # Compute effective dimensions
            rx = model.NewIntVar(min(w_val, h_val), max(w_val, h_val), f"rx_{i}")
            ry = model.NewIntVar(min(w_val, h_val), max(w_val, h_val), f"ry_{i}")
            
            # Link rotation to dimensions
            if r_flag:
                model.Add(rx == w_val).OnlyEnforceIf(rotated[i].Not())
                model.Add(rx == h_val).OnlyEnforceIf(rotated[i])
                model.Add(ry == h_val).OnlyEnforceIf(rotated[i].Not())
                model.Add(ry == w_val).OnlyEnforceIf(rotated[i])
            else:
                model.Add(rx == w_val)
                model.Add(ry == h_val)
            
            # Container constraints
            model.Add(x[i] + rx <= W).OnlyEnforceIf(placed[i])
            model.Add(y[i] + ry <= H).OnlyEnforceIf(placed[i])
            
            # Create end variables for interval
            x_end = model.NewIntVar(0, W, f"x_end_{i}")
            model.Add(x_end == x[i] + rx).OnlyEnforceIf(placed[i])
            x_interval = model.NewOptionalIntervalVar(
                x[i], rx, x_end, placed[i], f"x_interval_{i}"
            )
            
            y_end = model.NewIntVar(0, H, f"y_end_{i}")
            model.Add(y_end == y[i] + ry).OnlyEnforceIf(placed[i])
            y_interval = model.NewOptionalIntervalVar(
                y[i], ry, y_end, placed[i], f"y_interval_{i}"
            )
            
            x_intervals.append(x_interval)
            y_intervals.append(y_interval)
        
        # Efficient 2D no-overlap constraint
        model.AddNoOverlap2D(x_intervals, y_intervals)
        # Symmetry breaking: Order rectangles by position
        for i in range(n - 1):
            # If both placed, rectangle i should be left of or below rectangle j
            both_placed = model.NewBoolVar(f"both_placed_{i}")
            model.AddBoolAnd([placed[i], placed[i+1]]).OnlyEnforceIf(both_placed)
            model.AddBoolOr([placed[i].Not(), placed[i+1].Not()]).OnlyEnforceIf(both_placed.Not())
            
            # Left of or same x with lower y
            left_or_below = model.NewBoolVar(f"left_or_below_{i}")
            model.Add(x[i] <= x[i+1]).OnlyEnforceIf([both_placed, left_or_below])
            model.Add(y[i] <= y[i+1]).OnlyEnforceIf([both_placed, left_or_below.Not()])
            model.AddBoolOr([left_or_below, left_or_below.Not()]).OnlyEnforceIf(both_placed)
        # Additional symmetry breaking for identical rectangles
        for i in range(n):
            for j in range(i + 1, n):
                if sorted_rectangles[i][0] == sorted_rectangles[j][0] and \
                   sorted_rectangles[i][1] == sorted_rectangles[j][1] and \
                   sorted_rectangles[i][2] == sorted_rectangles[j][2]:
                    # If both are placed, rectangle i must be placed at a lower or equal x-coordinate
                    model.Add(x[i] <= x[j]).OnlyEnforceIf([placed[i], placed[j]])
        # Objective: maximize number of placed rectangles
        model.Maximize(sum(placed))
        
        # Solve with optimized parameters
        solver = cp_model.CpSolver()
        time_limit = kwargs.get('time_limit', 900.0)
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = 8
        solver.parameters.log_search_progress = False
        solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
        solver.parameters.linearization_level = 2
        
        status = solver.Solve(model)
        solution = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i in range(n):
                if solver.Value(placed[i]):
                    original_idx = sorted_indices[i]
                    solution.append((
                        original_idx,
                        solver.Value(x[i]),
                        solver.Value(y[i]),
                        bool(solver.Value(rotated[i]))
                    ))
        return solution