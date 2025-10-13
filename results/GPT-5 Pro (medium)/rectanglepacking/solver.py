from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the rectangle packing problem:
        Input: (W, H, [(w, h, rotatable), ...])
        Output: list of (index, x, y, rotated)
        """
        try:
            # Support for named instance types (if provided by harness)
            W = int(problem.container_width)
            H = int(problem.container_height)
            rects_raw = [(int(r.width), int(r.height), bool(r.rotatable)) for r in problem.rectangles]
        except Exception:
            # Tuple input (W, H, rectangles)
            W, H, rects = problem
            W, H = int(W), int(H)
            rects_raw = [(int(w), int(h), bool(r)) for (w, h, r) in rects]

        # Quick exit: no rectangles or zero-size container
        if W <= 0 or H <= 0 or not rects_raw:
            return []

        # Import OR-Tools locally to avoid import overhead on module import
        from ortools.sat.python import cp_model

        n = len(rects_raw)

        # Preprocess rectangles: filter out those that can never fit and deduplicate orientation if square.
        valid_indices: List[int] = []
        orientations: List[List[Tuple[int, int, bool]]] = []  # per kept rect: list of (w, h, rotated_flag)
        areas: List[int] = []

        for idx, (w, h, rot) in enumerate(rects_raw):
            opts: List[Tuple[int, int, bool]] = []
            if w <= W and h <= H:
                opts.append((w, h, False))
            if rot and h <= W and w <= H:
                # Avoid duplicating identical orientation (square or equal dims)
                if w != h or (w == h and not opts):
                    opts.append((h, w, True))
            if not opts:
                # Cannot place this rectangle at all
                continue
            valid_indices.append(idx)
            orientations.append(opts)
            areas.append(w * h)  # area independent of rotation

        if not valid_indices:
            return []

        model = cp_model.CpModel()

        # Variables
        placed_vars: List[cp_model.IntVar] = []
        x_intervals = []
        y_intervals = []

        # Store per-rect per-orientation vars to extract solution
        # Each entry: list of tuples (presence, x_start, y_start, rotated_flag)
        per_rect_data: List[List[Tuple[cp_model.IntVar, cp_model.IntVar, cp_model.IntVar, bool]]] = []

        # Build variables and intervals
        for ridx, opts in enumerate(orientations):
            placed = model.NewBoolVar(f"placed_{ridx}")
            placed_vars.append(placed)

            per_opts_vars: List[Tuple[cp_model.IntVar, cp_model.IntVar, cp_model.IntVar, bool]] = []
            presences: List[cp_model.IntVar] = []

            for oidx, (w, h, rotated) in enumerate(opts):
                # Start coordinates domains tightened to ensure end within container
                x_start = model.NewIntVar(0, W - w, f"x_{ridx}_{oidx}")
                y_start = model.NewIntVar(0, H - h, f"y_{ridx}_{oidx}")
                x_end = model.NewIntVar(0, W, f"x_end_{ridx}_{oidx}")
                y_end = model.NewIntVar(0, H, f"y_end_{ridx}_{oidx}")
                pres = model.NewBoolVar(f"pres_{ridx}_{oidx}")

                x_int = model.NewOptionalIntervalVar(x_start, w, x_end, pres, f"x_int_{ridx}_{oidx}")
                y_int = model.NewOptionalIntervalVar(y_start, h, y_end, pres, f"y_int_{ridx}_{oidx}")

                x_intervals.append(x_int)
                y_intervals.append(y_int)

                per_opts_vars.append((pres, x_start, y_start, rotated))
                presences.append(pres)

            # At most one orientation can be chosen; placed equals sum of presences
            model.Add(sum(presences) == placed)

            per_rect_data.append(per_opts_vars)

        # Global non-overlap constraint
        model.AddNoOverlap2D(x_intervals, y_intervals)

        # Area-based upper bound to help pruning
        total_area = W * H
        model.Add(sum(areas[i] * placed_vars[i] for i in range(len(placed_vars))) <= total_area)

        # Objective: maximize the number of placed rectangles
        model.Maximize(sum(placed_vars))

        # Solver parameters
        solver = cp_model.CpSolver()
        # Respect user-provided time limit if any; otherwise keep modest bound
        time_limit = kwargs.get("time_limit", None)
        if isinstance(time_limit, (int, float)) and time_limit > 0:
            solver.parameters.max_time_in_seconds = float(time_limit)
        else:
            solver.parameters.max_time_in_seconds = 30.0  # generous but typically not reached
        solver.parameters.num_search_workers = kwargs.get("num_workers", 8)
        solver.parameters.log_search_progress = False

        status = solver.Solve(model)
        solution: List[Tuple[int, int, int, bool]] = []

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for ridx, opt_vars in enumerate(per_rect_data):
                chosen = None
                xs = ys = 0
                rot = False
                for (pres, x_start, y_start, rotated) in opt_vars:
                    if solver.Value(pres):
                        xs = int(solver.Value(x_start))
                        ys = int(solver.Value(y_start))
                        rot = bool(rotated)
                        chosen = True
                        break
                if chosen:
                    orig_idx = valid_indices[ridx]
                    solution.append((orig_idx, xs, ys, rot))

        # Return placements; order does not matter for validation
        return solution