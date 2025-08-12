from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        # Unpack container dimensions and rectangles
        W, H, rectangles = problem
        N = len(rectangles)
        # Create CP-SAT model
        model = cp_model.CpModel()
        # Coordinate variables
        blx = [model.NewIntVar(0, W, f"blx{i}") for i in range(N)]
        bly = [model.NewIntVar(0, H, f"bly{i}") for i in range(N)]
        urx = [model.NewIntVar(0, W, f"urx{i}") for i in range(N)]
        ury = [model.NewIntVar(0, H, f"ury{i}") for i in range(N)]
        # Presence booleans per orientation
        pres_nr = [None] * N
        pres_r = [None] * N
        intervals_x = []
        intervals_y = []
        # Build optional intervals
        for i, (w, h, rotatable) in enumerate(rectangles):
            # No-rotation case
            pnr = model.NewBoolVar(f"pnr{i}")
            pres_nr[i] = pnr
            intervals_x.append(model.NewOptionalIntervalVar(blx[i], w, urx[i], pnr, f"ix_{i}_nr"))
            intervals_y.append(model.NewOptionalIntervalVar(bly[i], h, ury[i], pnr, f"iy_{i}_nr"))
            # Rotated case if allowed
            if rotatable:
                pr = model.NewBoolVar(f"pr{i}")
                pres_r[i] = pr
                intervals_x.append(model.NewOptionalIntervalVar(blx[i], h, urx[i], pr, f"ix_{i}_r"))
                intervals_y.append(model.NewOptionalIntervalVar(bly[i], w, ury[i], pr, f"iy_{i}_r"))
                # At most one orientation
                model.Add(pnr + pr <= 1)
            else:
                pres_r[i] = None
        # Non-overlap 2D constraint
        model.AddNoOverlap2D(intervals_x, intervals_y)
        # Objective: maximize number of placed rectangles
        obj_terms = []
        obj_terms.extend([p for p in pres_nr if p is not None])
        obj_terms.extend([p for p in pres_r if p is not None])
        model.Maximize(sum(obj_terms))
        # Solve model
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        # Extract placements
        placements = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i in range(N):
                if solver.Value(pres_nr[i]):
                    x = solver.Value(blx[i]); y = solver.Value(bly[i])
                    placements.append((i, x, y, False))
                elif pres_r[i] is not None and solver.Value(pres_r[i]):
                    x = solver.Value(blx[i]); y = solver.Value(bly[i])
                    placements.append((i, x, y, True))
        return placements