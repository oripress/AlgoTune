from __future__ import annotations

from typing import Any, List
import numpy as np
from scipy.spatial import ConvexHull

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        points = problem["points"]
        # Avoid unnecessary conversion if already a NumPy array with correct shape
        if isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 2:
            pts = points
        else:
            pts = np.asarray(points, dtype=float)

        n = pts.shape[0]

        # Tiny fast path for exactly three points
        if n == 3:
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            x2, y2 = pts[2]
            cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            hv = [0, 1, 2] if cross >= 0 else [0, 2, 1]
            return {"hull_vertices": hv, "hull_points": pts[hv].tolist()}

        # Lightweight Andrew's monotone chain for very small n to avoid Qhull overhead
        if n <= 8:
            x = pts[:, 0]
            y = pts[:, 1]
            order = np.lexsort((y, x)).astype(np.int64)

            lower: List[int] = []
            la = lower.append
            lp = lower.pop
            for j in order:
                j = int(j)
                while len(lower) >= 2:
                    o = lower[-2]
                    a = lower[-1]
                    if (x[a] - x[o]) * (y[j] - y[o]) - (y[a] - y[o]) * (x[j] - x[o]) <= 0.0:
                        lp()
                    else:
                        break
                la(j)

            upper: List[int] = []
            ua = upper.append
            up = upper.pop
            for j in reversed(order):
                j = int(j)
                while len(upper) >= 2:
                    o = upper[-2]
                    a = upper[-1]
                    if (x[a] - x[o]) * (y[j] - y[o]) - (y[a] - y[o]) * (x[j] - x[o]) <= 0.0:
                        up()
                    else:
                        break
                ua(j)

            hv = lower[:-1] + upper[:-1]

            # Degenerate safeguards (rare for random input)
            if len(hv) < 3:
                if len(hv) == 2:
                    a, b = hv
                    third = None
                    for jj in order:
                        jj = int(jj)
                        if jj != a and jj != b:
                            third = jj
                            break
                    hv = [a, (third if third is not None else b), b]
                elif len(hv) == 1:
                    a = hv[0]
                    hv = [a, a, a]
                else:
                    picks: List[int] = []
                    for jj in order:
                        jj = int(jj)
                        if jj not in picks:
                            picks.append(jj)
                        if len(picks) == 3:
                            break
                    while len(picks) < 3:
                        picks.append(picks[-1] if picks else 0)
                    hv = picks

            return {"hull_vertices": hv, "hull_points": pts[hv].tolist()}

        # For larger n, rely on fast C-optimized SciPy ConvexHull
        hull = ConvexHull(pts)
        hv = hull.vertices.tolist()
        return {"hull_vertices": hv, "hull_points": pts[hv].tolist()}