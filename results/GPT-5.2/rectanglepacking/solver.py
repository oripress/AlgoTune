"""Fast exact solver for rectangle packing (maximize number of non-overlapping rectangles).

Output: list of (index, x, y, rotated).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ortools.sat.python import cp_model

class _Rect:
    __slots__ = ("w", "h", "rot")

    def __init__(self, w: int, h: int, rot: bool):
        self.w = w
        self.h = h
        self.rot = rot

class _Place:
    __slots__ = ("i", "x", "y", "rotated", "w", "h")

    def __init__(self, i: int, x: int, y: int, rotated: bool, w: int, h: int):
        self.i = i
        self.x = x
        self.y = y
        self.rotated = rotated
        self.w = w
        self.h = h

def _parse_instance(problem: Any) -> Tuple[int, int, List[_Rect]]:
    if isinstance(problem, tuple) and len(problem) == 3:
        W, H, rects = problem
        return int(W), int(H), [_Rect(int(w), int(h), bool(r)) for (w, h, r) in rects]

    W = int(problem.container_width)
    H = int(problem.container_height)
    out: List[_Rect] = []
    for r in problem.rectangles:
        out.append(_Rect(int(r.width), int(r.height), bool(r.rotatable)))
    return W, H, out

def _max_k_by_area(container_area: int, areas: List[int]) -> int:
    areas.sort()
    s = 0
    k = 0
    for a in areas:
        s += a
        if s > container_area:
            break
        k += 1
    return k

def _greedy_pack(W: int, H: int, rects: Sequence[_Rect], order: List[int]) -> List[_Place]:
    free: List[Tuple[int, int, int, int]] = [(0, 0, W, H)]
    placed: List[_Place] = []

    def prune_free() -> None:
        nonlocal free
        out: List[Tuple[int, int, int, int]] = []
        for i, (x1, y1, w1, h1) in enumerate(free):
            if w1 <= 0 or h1 <= 0:
                continue
            contained = False
            for j, (x2, y2, w2, h2) in enumerate(free):
                if i == j:
                    continue
                if x2 <= x1 and y2 <= y1 and x2 + w2 >= x1 + w1 and y2 + h2 >= y1 + h1:
                    contained = True
                    break
            if not contained:
                out.append((x1, y1, w1, h1))
        free = out

    for i in order:
        r = rects[i]
        opts: List[Tuple[int, int, bool]] = []
        if r.w <= W and r.h <= H:
            opts.append((r.w, r.h, False))
        if r.rot and r.h <= W and r.w <= H and r.w != r.h:
            opts.append((r.h, r.w, True))
        if not opts:
            continue

        best_score: Optional[int] = None
        best_data: Optional[Tuple[int, int, int, int, int, int, bool, int]] = None
        for fi, (fx, fy, fw, fh) in enumerate(free):
            for rw, rh, rot in opts:
                if rw <= fw and rh <= fh:
                    score = min(fw - rw, fh - rh) * 1000000 + max(fw - rw, fh - rh)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_data = (fx, fy, fw, fh, rw, rh, rot, fi)

        if best_data is None:
            continue

        fx, fy, fw, fh, rw, rh, rot, fi = best_data
        placed.append(_Place(i=i, x=fx, y=fy, rotated=rot, w=rw, h=rh))

        free.pop(fi)
        rem_w = fw - rw
        rem_h = fh - rh
        if rem_w > 0:
            free.append((fx + rw, fy, rem_w, rh))
        if rem_h > 0:
            free.append((fx, fy + rh, fw, rem_h))
        prune_free()

    return placed

class Solver:
    def __init__(self) -> None:
        self._cpu = max(1, os.cpu_count() or 1)

    def solve(self, problem, **kwargs) -> Any:
        W, H, rects = _parse_instance(problem)
        if not rects:
            return []

        container_area = W * H
        areas = [r.w * r.h for r in rects]

        placeable: List[int] = []
        for i, r in enumerate(rects):
            if (r.w <= W and r.h <= H) or (r.rot and r.h <= W and r.w <= H):
                placeable.append(i)
        if not placeable:
            return []

        greedy = _greedy_pack(W, H, rects, sorted(placeable, key=areas.__getitem__))

        if len(greedy) == len(placeable):
            return [(p.i, p.x, p.y, p.rotated) for p in greedy]

        area_ub = _max_k_by_area(container_area, [areas[i] for i in placeable])
        if len(greedy) >= area_ub:
            return [(p.i, p.x, p.y, p.rotated) for p in greedy]

        greedy2 = _greedy_pack(W, H, rects, sorted(placeable, key=lambda i: max(rects[i].w, rects[i].h)))
        if len(greedy2) > len(greedy):
            greedy = greedy2

        model = cp_model.CpModel()

        opt_i: List[int] = []
        opt_rot: List[bool] = []
        opt_w: List[int] = []
        opt_h: List[int] = []
        opt_p: List[cp_model.IntVar] = []
        opt_x: List[cp_model.IntVar] = []
        opt_y: List[cp_model.IntVar] = []
        x_intervals: List[cp_model.IntervalVar] = []
        y_intervals: List[cp_model.IntervalVar] = []

        rect_to_opts: Dict[int, List[int]] = {}
        rect_key: Dict[int, Tuple[Tuple[int, int], ...]] = {}

        for i in placeable:
            r = rects[i]
            opts: List[Tuple[int, int, bool]] = []
            if r.w <= W and r.h <= H:
                opts.append((r.w, r.h, False))
            if r.rot and r.h <= W and r.w <= H and r.w != r.h:
                opts.append((r.h, r.w, True))
            if not opts:
                continue

            rect_to_opts[i] = []
            rect_key[i] = tuple(sorted((ow, oh) for (ow, oh, _rot) in opts))

            for ow, oh, rot in opts:
                x = model.NewIntVar(0, W - ow, "")
                y = model.NewIntVar(0, H - oh, "")
                p = model.NewBoolVar("")
                xi = model.NewOptionalFixedSizeIntervalVar(x, ow, p, "")
                yi = model.NewOptionalFixedSizeIntervalVar(y, oh, p, "")

                idx = len(opt_p)
                rect_to_opts[i].append(idx)
                opt_i.append(i)
                opt_rot.append(rot)
                opt_w.append(ow)
                opt_h.append(oh)
                opt_p.append(p)
                opt_x.append(x)
                opt_y.append(y)
                x_intervals.append(xi)
                y_intervals.append(yi)

        # At most one orientation per rectangle + symmetry break on identical rectangles.
        rect_sum: Dict[int, Any] = {}
        for i, idxs in rect_to_opts.items():
            if len(idxs) > 1:
                model.Add(sum(opt_p[j] for j in idxs) <= 1)
            rect_sum[i] = sum(opt_p[j] for j in idxs)

        groups: Dict[Tuple[Tuple[int, int], ...], List[int]] = {}
        for i, k in rect_key.items():
            groups.setdefault(k, []).append(i)
        for g in groups.values():
            if len(g) > 1:
                g.sort()
                for a, b in zip(g, g[1:]):
                    model.Add(rect_sum[a] >= rect_sum[b])

        model.AddNoOverlap2D(x_intervals, y_intervals)
        model.Maximize(sum(opt_p))

        # Hints from greedy.
        for p in greedy:
            idxs = rect_to_opts.get(p.i)
            if not idxs:
                continue
            chosen: Optional[int] = None
            for j in idxs:
                if opt_rot[j] == p.rotated and opt_w[j] == p.w and opt_h[j] == p.h:
                    chosen = j
                    break
            if chosen is None:
                for j in idxs:
                    if opt_rot[j] == p.rotated:
                        chosen = j
                        break
            if chosen is None:
                continue
            model.AddHint(opt_p[chosen], 1)
            model.AddHint(opt_x[chosen], p.x)
            model.AddHint(opt_y[chosen], p.y)

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        solver.parameters.num_search_workers = min(8, self._cpu)
        solver.parameters.symmetry_level = 2
        solver.parameters.cp_model_presolve = True
        solver.parameters.linearization_level = 2

        status = solver.Solve(model)
        if status == cp_model.OPTIMAL:
            pass
        elif status == cp_model.FEASIBLE:
            if int(solver.ObjectiveValue()) != int(solver.BestObjectiveBound()):
                return []
        else:
            return []

        out: List[Tuple[int, int, int, bool]] = []
        for j in range(len(opt_p)):
            if solver.Value(opt_p[j]) == 1:
                out.append((opt_i[j], int(solver.Value(opt_x[j])), int(solver.Value(opt_y[j])), bool(opt_rot[j])))
        return out