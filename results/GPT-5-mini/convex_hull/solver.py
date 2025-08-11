from typing import Any, Dict, List
import numpy as np

# Prefer SciPy's Qhull implementation when available.
try:
    from scipy.spatial import ConvexHull  # type: ignore
    _HAS_SCIPY = True
except Exception:
    ConvexHull = None  # type: ignore
    _HAS_SCIPY = False

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        pts_in = problem.get("points", [])
        if pts_in is None or len(pts_in) == 0:
            return {"hull_vertices": [], "hull_points": []}

        n = len(pts_in)
        if n == 1:
            return {"hull_vertices": [0, 0, 0], "hull_points": [pts_in[0], pts_in[0], pts_in[0]]}
        if n == 2:
            p0, p1 = pts_in[0], pts_in[1]
            if float(p0[0]) == float(p1[0]) and float(p0[1]) == float(p1[1]):
                return {"hull_vertices": [0, 0, 0], "hull_points": [p0, p0, p0]}
            return {"hull_vertices": [0, 1, 1], "hull_points": [p0, p1, p1]}

        # Fast path: SciPy ConvexHull directly on the input points (avoids extra copies/unique)
        if _HAS_SCIPY:
            try:
                hull = ConvexHull(np.asarray(pts_in, dtype=np.float64))
                hv = hull.vertices.tolist()
                hull_points = [pts_in[i] for i in hv]
                return {"hull_vertices": hv, "hull_points": hull_points}
            except Exception:
                # If Qhull fails for this input, fall back to Python implementation
                pass

        # Pure-Python fallback: Andrew's monotone chain (concise and efficient)
        pts = [(float(p[0]), float(p[1]), i) for i, p in enumerate(pts_in)]
        pts.sort(key=lambda t: (t[0], t[1]))

        # Remove exact duplicates (keep first occurrence)
        unique = []
        last_x = last_y = None
        for x, y, idx in pts:
            if last_x is None or x != last_x or y != last_y:
                unique.append((x, y, idx))
                last_x, last_y = x, y

        m = len(unique)
        if m == 0:
            return {"hull_vertices": [], "hull_points": []}
        if m == 1:
            i0 = unique[0][2]
            return {"hull_vertices": [i0, i0, i0], "hull_points": [pts_in[i0], pts_in[i0], pts_in[i0]]}
        if m == 2:
            i0 = unique[0][2]; i1 = unique[1][2]
            return {"hull_vertices": [i0, i1, i1], "hull_points": [pts_in[i0], pts_in[i1], pts_in[i1]]}

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in unique:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(unique):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = lower[:-1] + upper[:-1]
        if len(hull) < 3:
            # Degenerate: fallback to first unique occurrences preserving input order
            seen = set()
            uniq_indices = []
            for x, y, idx in pts:
                key = (x, y)
                if key not in seen:
                    seen.add(key)
                    uniq_indices.append(idx)
            if len(uniq_indices) >= 3:
                hull_indices = uniq_indices[:3]
            elif len(uniq_indices) == 2:
                hull_indices = [uniq_indices[0], uniq_indices[1], uniq_indices[1]]
            elif len(uniq_indices) == 1:
                hull_indices = [uniq_indices[0], uniq_indices[0], uniq_indices[0]]
            else:
                return {"hull_vertices": [], "hull_points": []}
            hull_points = [pts_in[i] for i in hull_indices]
            return {"hull_vertices": hull_indices, "hull_points": hull_points}

        hull_indices = [int(p[2]) for p in hull]
        hull_points = [pts_in[i] for i in hull_indices]
        return {"hull_vertices": hull_indices, "hull_points": hull_points}