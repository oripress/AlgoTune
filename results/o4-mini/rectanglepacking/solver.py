class Solver:
    def solve(self, problem, **kwargs):
        # Parse input
        try:
            # tuple/list form
            W, H, rects = problem
            recs = [(i, w, h, r) for i, (w, h, r) in enumerate(rects)]
        except Exception:
            # custom Instance form
            W = problem.container_width
            H = problem.container_height
            recs = [(i, r.width, r.height, r.rotatable) 
                    for i, r in enumerate(problem.rectangles)]

        # Filter out those that cannot possibly fit in any orientation
        filtered = []
        for i, w, h, rot in recs:
            if (w <= W and h <= H) or (rot and h <= W and w <= H):
                filtered.append((i, w, h, rot))
        # Sort by area ascending for greedy
        filtered.sort(key=lambda x: x[1] * x[2])

        # Quick greedy packing to get a decent initial solution
        candidates = [(0, 0)]
        cand_set = {(0, 0)}
        placed_boxes = []   # (x,y,w,h)
        greedy_sol = {}     # i -> (x, y, rot)
        for idx, w, h, rot in filtered:
            placed = False
            # try each candidate in y,x order
            for x0, y0 in sorted(candidates, key=lambda p: (p[1], p[0])):
                orients = [(w, h, False)]
                if rot:
                    orients.append((h, w, True))
                for w_r, h_r, rflag in orients:
                    if x0 + w_r > W or y0 + h_r > H:
                        continue
                    # check overlap
                    ok = True
                    for (xx, yy, ww, hh) in placed_boxes:
                        if not (x0 + w_r <= xx or xx + ww <= x0 or
                                y0 + h_r <= yy or yy + hh <= y0):
                            ok = False
                            break
                    if not ok:
                        continue
                    # place here
                    placed_boxes.append((x0, y0, w_r, h_r))
                    greedy_sol[idx] = (x0, y0, rflag)
                    # add new candidates
                    p1 = (x0 + w_r, y0)
                    p2 = (x0, y0 + h_r)
                    if p1[0] <= W and p1[1] <= H and p1 not in cand_set:
                        candidates.append(p1); cand_set.add(p1)
                    if p2[0] <= W and p2[1] <= H and p2 not in cand_set:
                        candidates.append(p2); cand_set.add(p2)
                    placed = True
                    break
                if placed:
                    break

        # Build CP-SAT model with 2D no-overlap
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        n = len(filtered)
        x_vars = [model.NewIntVar(0, W, f"x{i}") for i in range(n)]
        y_vars = [model.NewIntVar(0, H, f"y{i}") for i in range(n)]
        rot_vars = [model.NewBoolVar(f"rot{i}") for i in range(n)]
        place_vars = [model.NewBoolVar(f"place{i}") for i in range(n)]
        w_eff = [model.NewIntVar(0, max(W, H), f"weff{i}") for i in range(n)]
        # End coordinates for intervals
        x_end = [model.NewIntVar(0, W, f"xend{i}") for i in range(n)]
        y_end = [model.NewIntVar(0, H, f"yend{i}") for i in range(n)]
        x_ints = []
        y_ints = []
        h_eff = [model.NewIntVar(0, H, f"heff{i}") for i in range(n)]
        x_ints = []
        y_ints = []

        # Add rectangle constraints
        for j, (orig_i, w, h, can_rot) in enumerate(filtered):
            # rotation feasibility
            if not can_rot:
                model.Add(rot_vars[j] == 0)
            # zero‐size when not placed
            model.Add(w_eff[j] == 0).OnlyEnforceIf(place_vars[j].Not())
            model.Add(h_eff[j] == 0).OnlyEnforceIf(place_vars[j].Not())
            # size when placed & not rotated
            model.Add(w_eff[j] == w).OnlyEnforceIf([place_vars[j], rot_vars[j].Not()])
            model.Add(h_eff[j] == h).OnlyEnforceIf([place_vars[j], rot_vars[j].Not()])
            # size when placed & rotated
            model.Add(w_eff[j] == h).OnlyEnforceIf([place_vars[j], rot_vars[j]])
            # inside container
            model.Add(x_vars[j] + w_eff[j] <= W)
            model.Add(w_eff[j] == h).OnlyEnforceIf([place_vars[j], rot_vars[j]])
            model.Add(h_eff[j] == w).OnlyEnforceIf([place_vars[j], rot_vars[j]])
            # link end coordinates
            model.Add(x_vars[j] + w_eff[j] == x_end[j])
            model.Add(y_vars[j] + h_eff[j] == y_end[j])
            # build intervals
            x_ints.append(model.NewIntervalVar(x_vars[j], w_eff[j], x_end[j], f"xi{j}"))
            y_ints.append(model.NewIntervalVar(y_vars[j], h_eff[j], y_end[j], f"yi{j}"))

        # No overlap in 2D
        model.AddNoOverlap2D(x_ints, y_ints)
        # Maximize number of placed rectangles
        model.Maximize(sum(place_vars))

        # Hint the greedy solution
        for j, (orig_i, _, _, _) in enumerate(filtered):
            if orig_i in greedy_sol:
                x0, y0, rflag = greedy_sol[orig_i]
                model.AddHint(place_vars[j], 1)
                model.AddHint(x_vars[j], x0)
                model.AddHint(y_vars[j], y0)
                model.AddHint(rot_vars[j], int(rflag))
            else:
                model.AddHint(place_vars[j], 0)
                model.AddHint(rot_vars[j], 0)

        # Solve
        solver = cp_model.CpSolver()
        # respect an optional time_limit kwarg (seconds)
        tl = kwargs.get("time_limit", None)
        if isinstance(tl, (int, float)) and tl > 0:
            solver.parameters.max_time_in_seconds = tl
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)

        # Extract solution
        result = []
        for j, (orig_i, _, _, _) in enumerate(filtered):
            if solver.Value(place_vars[j]):
                result.append((orig_i,
                               int(solver.Value(x_vars[j])),
                               int(solver.Value(y_vars[j])),
                               bool(solver.Value(rot_vars[j]))))
        return result