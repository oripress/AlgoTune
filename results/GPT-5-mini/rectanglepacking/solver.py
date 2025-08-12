import time
from typing import Any, List, Tuple

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve rectangle packing: return list of (index, x, y, rotated).
        Supports input as (W, H, [(w,h,rot), ...]) or object-like instances.
        """
        t0 = time.perf_counter()
        time_limit = float(kwargs.get("time_limit", 1.0))

        W, H, rects = self._parse_instance(problem)
        if W <= 0 or H <= 0 or not rects:
            return []

        # Keep only rectangles that can fit in some orientation
        filtered = []
        for r in rects:
            w, h, rot = r["width"], r["height"], r["rotatable"]
            if (w <= W and h <= H) or (rot and h <= W and w <= H):
                filtered.append(r)
        rects = filtered
        if not rects:
            return []

        # Several heuristic orderings to try
        order_keys = [
            lambda r: -r["area"],  # largest area first
            lambda r: -max(r["width"], r["height"]),  # largest side first
            lambda r: r["area"],  # smallest area first
            lambda r: (r["width"] + r["height"]),  # small perimeter first
        ]

        best_solution: List[Tuple[int, int, int, bool]] = []
        best_count = -1

        for keyfn in order_keys:
            if time.perf_counter() - t0 > time_limit:
                break
            ordering = sorted(rects, key=keyfn)
            sol = self._pack_bottom_left(W, H, ordering, t0, time_limit)
            if len(sol) > best_count:
                best_solution = sol
                best_count = len(sol)
                if best_count == len(rects):
                    break

        # Format final solution: unique indices, ints and bools
        final = []
        seen = set()
        for idx, x, y, rot in best_solution:
            if idx in seen:
                continue
            seen.add(idx)
            final.append((int(idx), int(x), int(y), bool(rot)))
        return final

    def _parse_instance(self, problem):
        """
        Parse instance into (W, H, rects) where rects is list of dicts:
        {index, width, height, rotatable, area}
        """
        W = H = None
        raw_rects = None
        try:
            if isinstance(problem, (list, tuple)) and len(problem) >= 3:
                W = int(problem[0])
                H = int(problem[1])
                raw_rects = problem[2]
        except Exception:
            raw_rects = None

        if raw_rects is None:
            # try object-like
            W = getattr(problem, "container_width", None)
            H = getattr(problem, "container_height", None)
            raw_rects = getattr(problem, "rectangles", None)
            if W is None or H is None or raw_rects is None:
                return 0, 0, []
            W = int(W)
            H = int(H)

        rects = []
        for i, r in enumerate(raw_rects):
            try:
                if isinstance(r, (list, tuple)):
                    w = int(r[0])
                    h = int(r[1])
                    rot = bool(r[2]) if len(r) > 2 else False
                else:
                    # object-like rectangle
                    if hasattr(r, "width") and hasattr(r, "height"):
                        w = int(getattr(r, "width"))
                        h = int(getattr(r, "height"))
                        rot = bool(getattr(r, "rotatable", getattr(r, "rot", False)))
                    else:
                        # fallback: sequence-like
                        w = int(r[0])
                        h = int(r[1])
                        rot = bool(r[2]) if len(r) > 2 else False
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue
            rects.append({"index": i, "width": w, "height": h, "rotatable": rot, "area": w * h})
        return int(W), int(H), rects

    def _pack_bottom_left(self, W: int, H: int, rects: List[dict], t0: float, time_limit: float):
        """
        Bottom-left greedy placement with anchor points.
        Returns list of (index, x, y, rotated).
        """
        placed: List[Tuple[int, int, int, int, int, bool]] = []  # idx, x, y, w, h, rot
        candidates: List[Tuple[int, int]] = [(0, 0)]

        def overlaps(px: int, py: int, w: int, h: int) -> bool:
            for (_, ox, oy, ow, oh, _) in placed:
                if not (px + w <= ox or ox + ow <= px or py + h <= oy or oy + oh <= py):
                    return True
            return False

        def fits(px: int, py: int, w: int, h: int) -> bool:
            if px < 0 or py < 0 or px + w > W or py + h > H:
                return False
            return not overlaps(px, py, w, h)

        def filter_candidates(cands: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            seen = set()
            out = []
            for (cx, cy) in cands:
                if cx < 0 or cy < 0 or cx >= W or cy >= H:
                    continue
                inside = False
                for (_, px, py, pw, ph, _) in placed:
                    if px <= cx < px + pw and py <= cy < py + ph:
                        inside = True
                        break
                if not inside and (cx, cy) not in seen:
                    seen.add((cx, cy))
                    out.append((cx, cy))
            return out

        for r in rects:
            if time.perf_counter() - t0 > time_limit:
                break
            idx = r["index"]
            w0, h0 = r["width"], r["height"]
            rot_allowed = r["rotatable"]
            placed_flag = False
            anchors = sorted(filter_candidates(candidates), key=lambda p: (p[1], p[0]))  # lowest y then lowest x
            for (cx, cy) in anchors:
                if placed_flag:
                    break
                orientations = [(w0, h0, False)]
                if rot_allowed and (w0 != h0):
                    orientations.append((h0, w0, True))
                for (w, h, rotf) in orientations:
                    if fits(cx, cy, w, h):
                        placed.append((idx, cx, cy, w, h, rotf))
                        placed_flag = True
                        # add new anchors: right and top of placed rectangle
                        candidates.append((cx + w, cy))
                        candidates.append((cx, cy + h))
                        break
        return [(p[0], p[1], p[2], p[5]) for p in placed]