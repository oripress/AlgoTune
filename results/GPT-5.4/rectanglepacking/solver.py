from __future__ import annotations

import os
from typing import Any

def _contained(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax >= bx and ay >= by and ax + aw <= bx + bw and ay + ah <= by + bh

class Solver:
    def __init__(self) -> None:
        try:
            from ortools.sat.python import cp_model
        except Exception:
            cp_model = None
        self.cp_model = cp_model
        self.num_workers = max(1, min(8, os.cpu_count() or 1))

    def _normalize_problem(
        self, problem: Any
    ) -> tuple[int, int, list[tuple[int, int, int, bool, bool, bool]]]:
        if hasattr(problem, "container_width"):
            width = int(problem.container_width)
            height = int(problem.container_height)
            rectangles = problem.rectangles
        else:
            width = int(problem[0])
            height = int(problem[1])
            rectangles = problem[2]

        items: list[tuple[int, int, int, bool, bool, bool]] = []
        for i, rect in enumerate(rectangles):
            if hasattr(rect, "width"):
                w = int(rect.width)
                h = int(rect.height)
                rot = bool(rect.rotatable)
            else:
                w = int(rect[0])
                h = int(rect[1])
                rot = bool(rect[2])

            fits_normal = w <= width and h <= height
            fits_rotated = rot and h <= width and w <= height
            if fits_normal or fits_rotated:
                items.append((i, w, h, rot, fits_normal, fits_rotated))
        return width, height, items

    def _split_free_rectangles(
        self,
        free_rectangles: list[tuple[int, int, int, int]],
        px: int,
        py: int,
        pw: int,
        ph: int,
    ) -> list[tuple[int, int, int, int]]:
        px2 = px + pw
        py2 = py + ph
        new_free: list[tuple[int, int, int, int]] = []

        for rx, ry, rw, rh in free_rectangles:
            rx2 = rx + rw
            ry2 = ry + rh
            if px2 <= rx or px >= rx2 or py2 <= ry or py >= ry2:
                new_free.append((rx, ry, rw, rh))
                continue

            if px > rx:
                new_free.append((rx, ry, px - rx, rh))
            if px2 < rx2:
                new_free.append((px2, ry, rx2 - px2, rh))
            if py > ry:
                new_free.append((rx, ry, rw, py - ry))
            if py2 < ry2:
                new_free.append((rx, py2, rw, ry2 - py2))

        filtered = [r for r in new_free if r[2] > 0 and r[3] > 0]
        pruned: list[tuple[int, int, int, int]] = []
        for i, rect_i in enumerate(filtered):
            keep = True
            for j, rect_j in enumerate(filtered):
                if i != j and _contained(rect_i, rect_j):
                    keep = False
                    break
            if keep:
                pruned.append(rect_i)
        return pruned

    def _pack_maxrects(
        self,
        width: int,
        height: int,
        items: list[tuple[int, int, int, bool, bool, bool]],
        order_mode: str,
    ) -> list[tuple[int, int, int, bool]]:
        if order_mode == "small_area":
            ordered = sorted(
                items,
                key=lambda t: (t[1] * t[2], max(t[1], t[2]), min(t[1], t[2]), t[0]),
            )
        elif order_mode == "small_max_side":
            ordered = sorted(
                items,
                key=lambda t: (max(t[1], t[2]), t[1] * t[2], min(t[1], t[2]), t[0]),
            )
        elif order_mode == "small_height":
            ordered = sorted(
                items,
                key=lambda t: (min(t[1], t[2]), t[1] * t[2], max(t[1], t[2]), t[0]),
            )
        elif order_mode == "large_area":
            ordered = sorted(
                items,
                key=lambda t: (-(t[1] * t[2]), -max(t[1], t[2]), -min(t[1], t[2]), t[0]),
            )
        else:
            ordered = sorted(
                items,
                key=lambda t: (-max(t[1], t[2]), -(t[1] * t[2]), -min(t[1], t[2]), t[0]),
            )

        free_rectangles: list[tuple[int, int, int, int]] = [(0, 0, width, height)]
        placements: list[tuple[int, int, int, bool]] = []

        for idx, w, h, _, fits_normal, fits_rotated in ordered:
            orientations = []
            if fits_normal:
                orientations.append((False, w, h))
            if fits_rotated and (h != w or not fits_normal):
                orientations.append((True, h, w))

            best_choice: tuple[int, int, int, int, bool] | None = None
            best_score: tuple[int, int, int, int, int] | None = None

            for fx, fy, fw, fh in free_rectangles:
                for rotated, ow, oh in orientations:
                    if ow <= fw and oh <= fh:
                        short_fit = min(fw - ow, fh - oh)
                        long_fit = max(fw - ow, fh - oh)
                        area_fit = fw * fh - ow * oh
                        score = (short_fit, long_fit, area_fit, fy, fx)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_choice = (fx, fy, ow, oh, rotated)

            if best_choice is None:
                continue

            px, py, pw, ph, rotated = best_choice
            placements.append((idx, px, py, rotated))
            free_rectangles = self._split_free_rectangles(free_rectangles, px, py, pw, ph)

        placements.sort()
        return placements

    def _heuristic_solution(
        self, width: int, height: int, items: list[tuple[int, int, int, bool, bool, bool]]
    ) -> list[tuple[int, int, int, bool]]:
        best: list[tuple[int, int, int, bool]] = []
        for mode in (
            "small_area",
            "small_max_side",
            "small_height",
            "large_area",
            "tall_first",
        ):
            candidate = self._pack_maxrects(width, height, items, mode)
            if len(candidate) > len(best):
                best = candidate
                if len(best) == len(items):
                    break
        return best

    def _area_upper_bound(
        self, width: int, height: int, items: list[tuple[int, int, int, bool, bool, bool]]
    ) -> int:
        areas = sorted(w * h for _, w, h, _, _, _ in items)
        capacity = width * height
        total = 0
        count = 0
        for area in areas:
            if total + area > capacity:
                break
            total += area
            count += 1
        return count

    def _build_model(
        self,
        width: int,
        height: int,
        items: list[tuple[int, int, int, bool, bool, bool]],
        heuristic: list[tuple[int, int, int, bool]],
        lower_bound: int,
        upper_bound: int,
        target_count: int | None,
        optimize: bool,
    ):
        cp_model = self.cp_model
        model = cp_model.CpModel()

        x_intervals = []
        y_intervals = []
        presence_vars = []
        coord_vars = []
        packed_exprs = []
        by_item: list[list[tuple[Any, Any, Any, bool]]] = []
        heuristic_map = {idx: (x, y, rotated) for idx, x, y, rotated in heuristic}
        identical_groups: dict[tuple[tuple[int, int], ...], list[int]] = {}

        new_opt_fixed = getattr(model, "NewOptionalFixedSizeIntervalVar", None)

        for item_pos, (idx, w, h, _, fits_normal, fits_rotated) in enumerate(items):
            options: list[tuple[bool, int, int]] = []
            if fits_normal:
                options.append((False, w, h))
            if fits_rotated and (h != w or not fits_normal):
                options.append((True, h, w))

            option_vars: list[tuple[Any, Any, Any, bool]] = []
            item_presence = []
            signature = []

            for opt_pos, (rotated, ow, oh) in enumerate(options):
                p = model.NewBoolVar(f"p_{item_pos}_{opt_pos}")
                x = model.NewIntVar(0, width - ow, f"x_{item_pos}_{opt_pos}")
                y = model.NewIntVar(0, height - oh, f"y_{item_pos}_{opt_pos}")

                if new_opt_fixed is not None:
                    xi = model.NewOptionalFixedSizeIntervalVar(
                        x, ow, p, f"xi_{item_pos}_{opt_pos}"
                    )
                    yi = model.NewOptionalFixedSizeIntervalVar(
                        y, oh, p, f"yi_{item_pos}_{opt_pos}"
                    )
                else:
                    xe = model.NewIntVar(ow, width, f"xe_{item_pos}_{opt_pos}")
                    ye = model.NewIntVar(oh, height, f"ye_{item_pos}_{opt_pos}")
                    model.Add(xe == x + ow)
                    model.Add(ye == y + oh)
                    xi = model.NewOptionalIntervalVar(
                        x, ow, xe, p, f"xi_{item_pos}_{opt_pos}"
                    )
                    yi = model.NewOptionalIntervalVar(
                        y, oh, ye, p, f"yi_{item_pos}_{opt_pos}"
                    )

                x_intervals.append(xi)
                y_intervals.append(yi)
                presence_vars.append(p)
                coord_vars.append(x)
                coord_vars.append(y)
                option_vars.append((p, x, y, rotated))
                item_presence.append(p)
                signature.append((ow, oh))

                if idx in heuristic_map:
                    hx, hy, hrot = heuristic_map[idx]
                    if hrot == rotated:
                        model.AddHint(p, 1)
                        model.AddHint(x, hx)
                        model.AddHint(y, hy)
                    else:
                        model.AddHint(p, 0)
                        model.AddHint(x, 0)
                        model.AddHint(y, 0)
                else:
                    model.AddHint(p, 0)
                    model.AddHint(x, 0)
                    model.AddHint(y, 0)

            if len(item_presence) > 1:
                model.Add(sum(item_presence) <= 1)

            packed_expr = sum(item_presence)
            packed_exprs.append(packed_expr)
            by_item.append(option_vars)
            identical_groups.setdefault(tuple(sorted(signature)), []).append(item_pos)

        if x_intervals:
            model.AddNoOverlap2D(x_intervals, y_intervals)

        model.Add(
            sum((items[i][1] * items[i][2]) * packed_exprs[i] for i in range(len(items)))
            <= width * height
        )

        if lower_bound > 0:
            model.Add(sum(packed_exprs) >= lower_bound)
        if upper_bound < len(items):
            model.Add(sum(packed_exprs) <= upper_bound)
        if target_count is not None:
            model.Add(sum(packed_exprs) >= target_count)

        for positions in identical_groups.values():
            if len(positions) > 1:
                for a, b in zip(positions, positions[1:]):
                    model.Add(packed_exprs[a] >= packed_exprs[b])

        if optimize:
            model.Maximize(sum(packed_exprs))

        if presence_vars:
            model.AddDecisionStrategy(
                presence_vars,
                cp_model.CHOOSE_FIRST,
                cp_model.SELECT_MAX_VALUE,
            )
        if coord_vars:
            model.AddDecisionStrategy(
                coord_vars,
                cp_model.CHOOSE_LOWEST_MIN,
                cp_model.SELECT_MIN_VALUE,
            )

        return model, by_item

    def _extract_solution(
        self,
        solver: Any,
        by_item: list[list[tuple[Any, Any, Any, bool]]],
        items: list[tuple[int, int, int, bool, bool, bool]],
    ) -> list[tuple[int, int, int, bool]]:
        result: list[tuple[int, int, int, bool]] = []
        for item_pos, option_vars in enumerate(by_item):
            idx = items[item_pos][0]
            for p, x, y, rotated in option_vars:
                if solver.BooleanValue(p):
                    result.append((idx, solver.Value(x), solver.Value(y), rotated))
                    break
        result.sort()
        return result

    def _make_solver(
        self, time_limit: float | None, n: int, first_solution: bool = False
    ) -> Any:
        solver = self.cp_model.CpSolver()
        params = solver.parameters
        params.log_search_progress = False
        if n <= 40:
            params.num_search_workers = 1
        elif n <= 100:
            params.num_search_workers = min(2, self.num_workers)
        else:
            params.num_search_workers = min(4, self.num_workers)
        params.random_seed = 0
        params.cp_model_presolve = True
        params.linearization_level = 0
        if first_solution:
            params.stop_after_first_solution = True
        if time_limit is not None:
            params.max_time_in_seconds = float(time_limit)
        return solver

    def _default_opt_time(self, n: int, gap: int) -> float:
        if n <= 20:
            return 0.6 if gap <= 2 else 0.4
        if n <= 50:
            return 1.4 if gap <= 2 else 0.9
        if n <= 100:
            return 3.0 if gap <= 2 else 1.8
        return 4.5

    def _default_target_time(self, n: int) -> float:
        if n <= 20:
            return 0.3
        if n <= 50:
            return 0.7
        if n <= 100:
            return 1.4
        return 2.0

    def _solve_cpsat(
        self,
        width: int,
        height: int,
        items: list[tuple[int, int, int, bool, bool, bool]],
        heuristic: list[tuple[int, int, int, bool]],
        lower_bound: int,
        upper_bound: int,
        time_limit: float | None,
    ) -> list[tuple[int, int, int, bool]]:
        cp_model = self.cp_model
        if cp_model is None:
            return heuristic

        n = len(items)
        gap = upper_bound - lower_bound

        if 0 < gap <= 2:
            per_try = (
                max(0.15, float(time_limit) / 2.0)
                if time_limit is not None
                else self._default_target_time(n)
            )
            infeasible_targets = 0
            for target in range(upper_bound, lower_bound, -1):
                model, by_item = self._build_model(
                    width,
                    height,
                    items,
                    heuristic,
                    lower_bound,
                    upper_bound,
                    target,
                    False,
                )
                solver = self._make_solver(per_try, n, first_solution=True)
                status = solver.Solve(model)
                if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    result = self._extract_solution(solver, by_item, items)
                    if len(result) >= target:
                        return result
                elif status == cp_model.INFEASIBLE:
                    infeasible_targets += 1

            if infeasible_targets == gap:
                return heuristic

        model, by_item = self._build_model(
            width,
            height,
            items,
            heuristic,
            lower_bound,
            upper_bound,
            None,
            True,
        )
        solver = self._make_solver(
            float(time_limit) if time_limit is not None else self._default_opt_time(n, gap),
            n,
        )
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return heuristic

        result = self._extract_solution(solver, by_item, items)
        if len(result) < len(heuristic):
            return heuristic
        return result

    def solve(self, problem, **kwargs) -> Any:
        width, height, items = self._normalize_problem(problem)
        if width <= 0 or height <= 0 or not items:
            return []

        heuristic = self._heuristic_solution(width, height, items)
        lower_bound = len(heuristic)
        upper_bound = self._area_upper_bound(width, height, items)

        if lower_bound >= upper_bound or lower_bound == len(items):
            return heuristic

        cp_items = sorted(
            items,
            key=lambda t: (t[1] * t[2], max(t[1], t[2]), min(t[1], t[2]), t[0]),
        )

        try:
            result = self._solve_cpsat(
                width,
                height,
                cp_items,
                heuristic,
                lower_bound,
                upper_bound,
                kwargs.get("time_limit"),
            )
        except Exception:
            return heuristic

        if len(result) < lower_bound:
            return heuristic
        return result