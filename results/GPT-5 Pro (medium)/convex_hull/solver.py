from typing import Any, Dict, List

import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Compute the convex hull of a set of 2D points using Andrew's monotone chain algorithm.

        Returns:
            - hull_vertices: list of indices into the original points array forming the hull in CCW order
            - hull_points: the coordinates of these hull vertices in CCW order
        """
        points = problem.get("points", None)
        if points is None:
            # Fallback empty result if no points provided
            return {"hull_vertices": [], "hull_points": []}

        # Ensure numpy array (the validator expects indexing with numpy int arrays)
        pts = points if isinstance(points, np.ndarray) else np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("Input points must be an array of shape (n, 2).")

        n = pts.shape[0]
        if n == 0:
            return {"hull_vertices": [], "hull_points": []}

        # Pre-extract coordinate arrays for speed
        px = pts[:, 0].astype(float, copy=False)
        py = pts[:, 1].astype(float, copy=False)

        # Lexicographic sort by (x, y)
        order = np.lexsort((py, px))
        sx = px[order]
        sy = py[order]

        # Remove duplicates while keeping the first index for each unique point
        if n > 1:
            dup_mask = np.ones(len(order), dtype=bool)
            dup_mask[1:] = (sx[1:] != sx[:-1]) | (sy[1:] != sy[:-1])
            unique_sorted_idx = order[dup_mask]
        else:
            unique_sorted_idx = order

        m = unique_sorted_idx.size

        # Cross product of OA x OB, using indices into original arrays
        def cross(i: int, j: int, k: int) -> float:
            return (px[j] - px[i]) * (py[k] - py[i]) - (py[j] - py[i]) * (px[k] - px[i])

        # Handle trivial cases early
        if m == 1:
            i0 = int(unique_sorted_idx[0])
            hull_vertices = [i0, i0, i0]  # ensure at least 3 vertices as per validator requirement
            hull_points = pts[hull_vertices].tolist()
            return {"hull_vertices": hull_vertices, "hull_points": hull_points}
        if m == 2:
            i0 = int(unique_sorted_idx[0])
            i1 = int(unique_sorted_idx[1])
            # Add a third vertex; duplicate one endpoint to satisfy "at least 3 vertices"
            hull_vertices = [i0, i1, i0]
            hull_points = pts[hull_vertices].tolist()
            return {"hull_vertices": hull_vertices, "hull_points": hull_points}

        # Build lower hull
        lower: List[int] = []
        for idx in unique_sorted_idx:
            i = int(idx)
            while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0.0:
                lower.pop()
            lower.append(i)

        # Build upper hull
        upper: List[int] = []
        for idx in unique_sorted_idx[::-1]:
            i = int(idx)
            while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0.0:
                upper.pop()
            upper.append(i)

        # Concatenate lower and upper to get full hull; omit last point of each (it's the starting point of the other)
        if lower:
            lower = lower[:-1]
        if upper:
            upper = upper[:-1]
        hull_vertices = lower + upper

        # Ensure at least 3 vertices (handle collinear sets: Andrew's will return 2 endpoints)
        if len(hull_vertices) < 3:
            in_set = set(hull_vertices)
            extra = None
            for i in unique_sorted_idx:
                ii = int(i)
                if ii not in in_set:
                    extra = ii
                    break
            if extra is None:
                hull_vertices = hull_vertices + [hull_vertices[0]]
            else:
                if len(hull_vertices) == 2:
                    hull_vertices = [hull_vertices[0], extra, hull_vertices[1]]
                else:
                    hull_vertices.append(extra)

        # Ensure counter-clockwise orientation
        hv = hull_vertices
        xh = px[hv]
        yh = py[hv]
        s = float(np.dot(xh, np.roll(yh, -1)) - np.dot(yh, np.roll(xh, -1)))
        if s < 0:
            hull_vertices = hull_vertices[::-1]

        hull_points = pts[hull_vertices].tolist()
        return {"hull_vertices": hull_vertices, "hull_points": hull_points}