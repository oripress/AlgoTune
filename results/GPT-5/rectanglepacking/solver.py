from __future__ import annotations

from typing import Any, List, Tuple, Optional

try:
    # Import inside top-level so compilation time isn't counted in runtime.
    from ortools.sat.python import cp_model
except Exception:  # pragma: no cover - in case ortools is unavailable in some envs
    cp_model = None  # type: ignore

class Solver:
    def _normalize_problem(self, problem: Any) -> Tuple[int, int, List[Tuple[int, int, bool]]]:
        """
        Accept both the tuple format (W, H, [(w, h, rot), ...]) and
        potential NamedTuple/Instance-like objects with attributes:
          - container_width
          - container_height
          - rectangles: iterable with .width, .height, .rotatable
        """
        # Tuple/list format
        if isinstance(problem, (list, tuple)) and len(problem) == 3:
            W, H, rects = problem
            norm_rects: List[Tuple[int, int, bool]] = []
            for r in rects:
                # r can be (w, h, rot) or an object
                if isinstance(r, (list, tuple)) and len(r) >= 3:
                    w, h, rot = int(r[0]), int(r[1]), bool(r[2])
                else:
                    w, h = int(getattr(r, "width")), int(getattr(r, "height"))
                    rot = bool(getattr(r, "rotatable"))
                norm_rects.append((w, h, rot))
            return int(W), int(H), norm_rects

        # Instance-like object
        W = int(getattr(problem, "container_width"))
        H = int(getattr(problem, "container_height"))
        rects_obj = getattr(problem, "rectangles")
        norm_rects = []
        for r in rects_obj:
            w, h = int(getattr(r, "width")), int(getattr(r, "height"))
            rot = bool(getattr(r, "rotatable"))
            norm_rects.append((w, h, rot))
        return W, H, norm_rects

    @staticmethod
    def _guillotine_greedy(W: int, H: int, rects: List[Tuple[int, int, bool]]) -> List[Tuple[int, int, int, bool]]:
        """
        Simple greedy guillotine split heuristic to produce a fast feasible solution
        (placement list) to warm-start CP-SAT. Not necessarily optimal.
        Returns list of (index, x, y, rotated).
        """
        if W <= 0 or H <= 0 or not rects:
            return []

        # Active rectangles with index
        items = [(i, w, h, r) for i, (w, h, r) in enumerate(rects)]

        # Filter out impossible rectangles immediately
        def can_fit_any(w: int, h: int, rot: bool) -> bool:
            if w <= W and h <= H:
                return True
            if rot and h <= W and w <= H:
                return True
            return False

        items = [it for it in items if can_fit_any(it[1], it[2], it[3])]
        if not items:
            return []

        # Sort by area descending; tie-break by max side, then width
        items.sort(key=lambda t: (t[1] * t[2], max(t[1], t[2]), t[1]), reverse=True)

        # Free spaces: list of (x, y, w, h), disjoint partitions, start with the container
        free_spaces: List[Tuple[int, int, int, int]] = [(0, 0, W, H)]
        placements: List[Tuple[int, int, int, bool]] = []

        for idx, w, h, rot in items:
            best_choice: Optional[Tuple[int, int, int, int, int, int, bool, int]] = None
            # Stores (wasted_area, y, x, fs_w, fs_h, fs_index, rotated, orientation_id)
            # orientation_id: 0 -> (w,h), 1 -> (h,w)

            for fs_index, (fx, fy, fw, fh) in enumerate(free_spaces):
                # try not rotated
                if w <= fw and h <= fh:
                    wasted = fw * fh - w * h
                    cand = (wasted, fy, fx, fw, fh, fs_index, False, 0)
                    if (best_choice is None) or (cand < best_choice):
                        best_choice = cand
                # try rotated
                if rot and h <= fw and w <= fh:
                    wasted = fw * fh - h * w
                    cand = (wasted, fy, fx, fw, fh, fs_index, True, 1)
                    if (best_choice is None) or (cand < best_choice):
                        best_choice = cand

            if best_choice is None:
                continue  # cannot place this rectangle

            # place rectangle at bottom-left of chosen space
            _, fy, fx, fw, fh, fs_index, rotated, ori = best_choice
            pw, ph = (h, w) if rotated else (w, h)
            px, py = fx, fy

            placements.append((idx, px, py, rotated))

            # Split the free space guillotine-style into two non-overlapping spaces
            # Remove used free space
            del free_spaces[fs_index]
            # Right space: to the right of the placed rectangle (covering its height)
            right_w = fw - pw
            right_h = ph
            if right_w > 0 and right_h > 0:
                free_spaces.append((px + pw, py, right_w, right_h))
            # Top space: above the placed rectangle (full width of original free space)
            top_w = fw
            top_h = fh - ph
            if top_h > 0 and top_w > 0:
                free_spaces.append((px, py + ph, top_w, top_h))

        return placements

    def solve(self, problem, **kwargs) -> Any:
        """
        Rectangle packing: maximize number of packed rectangles within a container.
        Input: (W, H, [(w, h, rot), ...])
        Output: list of (index, x, y, rotated)
        """
        if cp_model is None:
            # Fallback: return empty if no solver available
            return []

        W, H, rects = self._normalize_problem(problem)

        # Short-circuit trivial cases
        if W <= 0 or H <= 0 or not rects:
            return []

        # Pre-filter rectangles that can't fit in any orientation
        def can_fit_any(w: int, h: int, rot: bool) -> bool:
            return (w <= W and h <= H) or (rot and h <= W and w <= H)

        active_indices: List[int] = []
        active_rects: List[Tuple[int, int, bool]] = []
        for i, (w, h, r) in enumerate(rects):
            if can_fit_any(w, h, r):
                active_indices.append(i)
                active_rects.append((w, h, r))
        n = len(active_rects)
        if n == 0:
            return []

        # Try a quick greedy to produce a strong initial feasible solution
        greedy_solution = self._guillotine_greedy(W, H, active_rects)
        # Map to dict for fast lookup
        greedy_map = {idx: (x, y, rot) for (idx, x, y, rot) in greedy_solution}

        # Build CP-SAT model
        model = cp_model.CpModel()

        # Variables
        x_starts: List[cp_model.IntVar] = []
        y_starts: List[cp_model.IntVar] = []
        placed: List[cp_model.BoolVar] = []
        rotated_vars: List[Optional[cp_model.BoolVar]] = []
        # For optional orientations
        orient0_presence: List[Optional[cp_model.BoolVar]] = []
        orient1_presence: List[Optional[cp_model.BoolVar]] = []

        x_intervals: List[cp_model.IntervalVar] = []
        y_intervals: List[cp_model.IntervalVar] = []

        for i, (w, h, rot) in enumerate(active_rects):
            # Start domains - we can tighten bounds to reduce search
            if rot:
                min_side = min(w, h)
                xs = model.NewIntVar(0, max(0, W - min_side), f"x_{i}")
                ys = model.NewIntVar(0, max(0, H - min_side), f"y_{i}")
            else:
                xs = model.NewIntVar(0, max(0, W - w), f"x_{i}")
                ys = model.NewIntVar(0, max(0, H - h), f"y_{i}")

            p = model.NewBoolVar(f"p_{i}")
            x_starts.append(xs)
            y_starts.append(ys)
            placed.append(p)

            if not rot:
                # Not rotatable: single optional interval with presence == placed
                rotated_vars.append(None)
                orient0_presence.append(None)
                orient1_presence.append(None)

                x_end = model.NewIntVar(0, W, f"x_end_{i}")
                y_end = model.NewIntVar(0, H, f"y_end_{i}")
                xi = model.NewOptionalIntervalVar(xs, w, x_end, p, f"x_int_{i}")
                yi = model.NewOptionalIntervalVar(ys, h, y_end, p, f"y_int_{i}")
                x_intervals.append(xi)
                y_intervals.append(yi)
            else:
                # Rotatable: two orientation intervals, exactly one if placed
                r = model.NewBoolVar(f"rot_{i}")
                rotated_vars.append(r)
                b0 = model.NewBoolVar(f"b0_{i}")  # not rotated
                b1 = model.NewBoolVar(f"b1_{i}")  # rotated
                orient0_presence.append(b0)
                orient1_presence.append(b1)

                # Link presence to placed and rotated
                # placed == b0 + b1, rotated == b1
                model.Add(p == b0 + b1)
                model.Add(r == b1)

                # Orientation 0 (w,h)
                x_end0 = model.NewIntVar(0, W, f"x_end0_{i}")
                y_end0 = model.NewIntVar(0, H, f"y_end0_{i}")
                xi0 = model.NewOptionalIntervalVar(xs, w, x_end0, b0, f"x_int0_{i}")
                yi0 = model.NewOptionalIntervalVar(ys, h, y_end0, b0, f"y_int0_{i}")
                x_intervals.append(xi0)
                y_intervals.append(yi0)

                # Orientation 1 (h,w)
                x_end1 = model.NewIntVar(0, W, f"x_end1_{i}")
                y_end1 = model.NewIntVar(0, H, f"y_end1_{i}")
                xi1 = model.NewOptionalIntervalVar(xs, h, x_end1, b1, f"x_int1_{i}")
                yi1 = model.NewOptionalIntervalVar(ys, w, y_end1, b1, f"y_int1_{i}")
                x_intervals.append(xi1)
                y_intervals.append(yi1)

        # No-overlap constraint across all intervals
        model.AddNoOverlap2D(x_intervals, y_intervals)

        # Objective: maximize number of placed rectangles
        model.Maximize(sum(placed))

        # Hints from greedy solution
        for i, (w, h, rot) in enumerate(active_rects):
            xs = x_starts[i]
            ys = y_starts[i]
            p = placed[i]
            hint = greedy_map.get(i, None)
            if hint is None:
                model.AddHint(p, 0)
                continue

            gx, gy, grot = hint
            model.AddHint(p, 1)
            # Respect bounds just in case
            gx = max(0, min(gx, W))
            gy = max(0, min(gy, H))
            model.AddHint(xs, gx)
            model.AddHint(ys, gy)
            if rot:
                r = rotated_vars[i]
                b0 = orient0_presence[i]
                b1 = orient1_presence[i]
                if r is not None and b0 is not None and b1 is not None:
                    model.AddHint(r, 1 if grot else 0)
                    model.AddHint(b1, 1 if grot else 0)
                    model.AddHint(b0, 0 if grot else 1)

        # Solve
        solver = cp_model.CpSolver()
        # Speed settings
        solver.parameters.log_search_progress = False
        # Allow multi-threading for speed; 0 lets CP-SAT decide (use available cores)
        solver.parameters.num_search_workers = 8

        # Optional time limit from kwargs
        time_limit = kwargs.get("time_limit", None)
        if time_limit is not None:
            try:
                solver.parameters.max_time_in_seconds = float(time_limit)
            except Exception:
                pass  # ignore invalid time_limit

        status = solver.Solve(model)

        solution: List[Tuple[int, int, int, bool]] = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i, (w, h, rot) in enumerate(active_rects):
                if solver.Value(placed[i]) == 0:
                    continue
                x = solver.Value(x_starts[i])
                y = solver.Value(y_starts[i])
                rotated = False
                if rot:
                    r = rotated_vars[i]
                    rotated = solver.Value(r) == 1 if r is not None else False
                orig_idx = active_indices[i]
                solution.append((orig_idx, x, y, rotated))
        else:
            # Fallback to greedy if for any reason solver failed (very unlikely)
            # Map back greedy indices to original indices
            for (i, x, y, rot) in greedy_solution:
                solution.append((active_indices[i], x, y, rot))

        return solution