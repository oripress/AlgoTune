import collections
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        W, H, rectangles = problem
        n = len(rectangles)
        
        # Add original index and calculate area
        rects_with_index = []
        for i, (w, h, r) in enumerate(rectangles):
            rects_with_index.append({'w': w, 'h': h, 'r': r, 'id': i, 'area': w*h})
            
        # Sort by area ascending (Smallest Area First)
        sorted_rects = sorted(rects_with_index, key=lambda x: x['area'])
        
        # Calculate Area-based Upper Bound
        total_area = 0
        max_items = 0
        container_area = W * H
        for r in sorted_rects:
            if total_area + r['area'] <= container_area:
                total_area += r['area']
                max_items += 1
            else:
                break
        
        # Greedy Heuristic (Bottom-Left)
        
        def run_greedy(sorted_rects_input):
            placed_rects = [] # List of (x, y, w, h)
            current_sol = []
            
            # Candidate coordinates
            xs = {0}
            ys = {0}
            
            for r in sorted_rects_input:
                # Try to place r
                found = False
                
                w, h = r['w'], r['h']
                rotatable = r['r']
                
                orientations = [(w, h, False)]
                if rotatable and w != h:
                    orientations.append((h, w, True))
                
                sorted_ys = sorted(list(ys))
                sorted_xs = sorted(list(xs))
                
                for y_pos in sorted_ys:
                    for x_pos in sorted_xs:
                        for (cw, ch, is_rot) in orientations:
                            if x_pos + cw > W or y_pos + ch > H:
                                continue
                            
                            # Check overlap
                            overlap = False
                            for (px, py, pw, ph) in placed_rects:
                                if not (x_pos + cw <= px or px + pw <= x_pos or y_pos + ch <= py or py + ph <= y_pos):
                                    overlap = True
                                    break
                            
                            if not overlap:
                                placed_rects.append((x_pos, y_pos, cw, ch))
                                current_sol.append((r['id'], x_pos, y_pos, is_rot))
                                
                                if x_pos + cw < W: xs.add(x_pos + cw)
                                if y_pos + ch < H: ys.add(y_pos + ch)
                                
                                found = True
                                break
                        if found: break
                    if found: break
            return current_sol

        # Strategy 1: Smallest Area First (Primary)
        greedy_solution = run_greedy(sorted_rects)
        greedy_count = len(greedy_solution)
        
        if greedy_count == max_items:
            return greedy_solution

        # Strategy 2: Smallest Width First
        sorted_rects_w = sorted(rects_with_index, key=lambda x: x['w'])
        sol_w = run_greedy(sorted_rects_w)
        if len(sol_w) > greedy_count:
            greedy_solution = sol_w
            greedy_count = len(sol_w)
        
        if greedy_count == max_items:
            return greedy_solution

        # Strategy 3: Smallest Height First
        sorted_rects_h = sorted(rects_with_index, key=lambda x: x['h'])
        sol_h = run_greedy(sorted_rects_h)
        if len(sol_h) > greedy_count:
            greedy_solution = sol_h
            greedy_count = len(sol_h)
            
        if greedy_count == max_items:
            return greedy_solution
            
        # Strategy 4: Max Dimension First (Longest side) - maybe good for packing
        sorted_rects_max = sorted(rects_with_index, key=lambda x: max(x['w'], x['h']))
        sol_max = run_greedy(sorted_rects_max)
        if len(sol_max) > greedy_count:
            greedy_solution = sol_max
            greedy_count = len(sol_max)
            
        if greedy_count == max_items:
            return greedy_solution
            return greedy_solution
            
        # If not optimal, use CP-SAT
        model = cp_model.CpModel()
        
        x = []
        y = []
        rotated = []
        placed = []
        x_intervals = []
        y_intervals = []
        
        valid_indices = []

        for i, (w, h, r) in enumerate(rectangles):
            p = model.NewBoolVar(f'placed_{i}')
            placed.append(p)
            
            rot = model.NewBoolVar(f'rotated_{i}')
            rotated.append(rot)
            
            fits_normal = (w <= W and h <= H)
            fits_rotated = (r and h <= W and w <= H)
            
            if not fits_normal and not fits_rotated:
                model.Add(p == 0)
                # Dummy variables
                x.append(model.NewIntVar(0, 0, f'x_{i}'))
                y.append(model.NewIntVar(0, 0, f'y_{i}'))
                x_intervals.append(model.NewOptionalIntervalVar(0, 1, 1, p, f'x_int_{i}'))
                y_intervals.append(model.NewOptionalIntervalVar(0, 1, 1, p, f'y_int_{i}'))
                continue
            
            valid_indices.append(i)

            if not r:
                model.Add(rot == 0)
            elif w == h:
                model.Add(rot == 0)
            else:
                if not fits_normal:
                    model.Add(rot == 1).OnlyEnforceIf(p)
                if not fits_rotated:
                    model.Add(rot == 0).OnlyEnforceIf(p)

            current_w = model.NewIntVar(0, W, f'w_{i}')
            current_h = model.NewIntVar(0, H, f'h_{i}')
            
            # Fix variables if not placed
            model.Add(current_w == 0).OnlyEnforceIf(p.Not())
            model.Add(current_h == 0).OnlyEnforceIf(p.Not())
            model.Add(rot == 0).OnlyEnforceIf(p.Not())
            
            # Dimensions if placed
            model.Add(current_w == w).OnlyEnforceIf([p, rot.Not()])
            model.Add(current_h == h).OnlyEnforceIf([p, rot.Not()])
            if r and w != h and fits_rotated:
                 model.Add(current_w == h).OnlyEnforceIf([p, rot])
                 model.Add(current_h == w).OnlyEnforceIf([p, rot])

            xi = model.NewIntVar(0, W, f'x_{i}')
            yi = model.NewIntVar(0, H, f'y_{i}')
            x.append(xi)
            y.append(yi)
            
            model.Add(xi == 0).OnlyEnforceIf(p.Not())
            model.Add(yi == 0).OnlyEnforceIf(p.Not())

            x_end = model.NewIntVar(0, W, f'x_end_{i}')
            y_end = model.NewIntVar(0, H, f'y_end_{i}')
            
            x_int = model.NewOptionalIntervalVar(xi, current_w, x_end, p, f'x_int_{i}')
            y_int = model.NewOptionalIntervalVar(yi, current_h, y_end, p, f'y_int_{i}')
            
            x_intervals.append(x_int)
            y_intervals.append(y_int)

        model.AddNoOverlap2D(x_intervals, y_intervals)
        
        # Symmetry breaking
        rect_signatures = collections.defaultdict(list)
        for i in valid_indices:
            w, h, r = rectangles[i]
            if r:
                dims = tuple(sorted((w, h)))
                sig = (dims[0], dims[1], True)
            else:
                sig = (w, h, False)
            rect_signatures[sig].append(i)
            
        for sig, indices in rect_signatures.items():
            indices.sort()
            if len(indices) > 1:
                for k in range(len(indices) - 1):
                    i_curr = indices[k]
                    i_next = indices[k+1]
                    model.Add(placed[i_curr] >= placed[i_next])
        
        # Bounds
        model.Add(sum(placed) <= max_items)
        model.Add(sum(placed) >= greedy_count)
        
        # Knapsack Constraint (Area)
        rect_areas = [w*h for w, h, r in rectangles]
        model.Add(sum(placed[i] * rect_areas[i] for i in range(n)) <= W * H)
        
        # Hints from greedy solution
        for (idx, gx, gy, grot) in greedy_solution:
            model.AddHint(placed[idx], 1)
            model.AddHint(x[idx], gx)
            model.AddHint(y[idx], gy)
            model.AddHint(rotated[idx], int(grot))

        # Objective
        model.Maximize(sum(placed))
        
        # Search Strategy
        # Prioritize placing smaller area rectangles
        sorted_indices_by_area = sorted(valid_indices, key=lambda i: rect_areas[i])
        sorted_placed_vars = [placed[i] for i in sorted_indices_by_area]
        model.AddDecisionStrategy(sorted_placed_vars, cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE)
        
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        
        # Adjust time limit based on gap
        if max_items - greedy_count <= 1:
             solver.parameters.max_time_in_seconds = 5.0
        else:
             solver.parameters.max_time_in_seconds = 60.0
        
        status = solver.Solve(model)
        
        solution = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i in range(n):
                if solver.Value(placed[i]):
                    solution.append((
                        i,
                        solver.Value(x[i]),
                        solver.Value(y[i]),
                        bool(solver.Value(rotated[i]))
                    ))
        
        if not solution:
            return greedy_solution
            
        return solution
        status = solver.Solve(model)
        
        solution = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i in range(n):
                if solver.Value(placed[i]):
                    solution.append((
                        i,
                        solver.Value(x[i]),
                        solver.Value(y[i]),
                        bool(solver.Value(rotated[i]))
                    ))
        return solution