from typing import Any, List
import numpy as np
from scipy.spatial import Delaunay

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute Delaunay triangulation using SciPy for the given set of 2D points.
        Optimized fast paths for very small point sets using vectorized checks (n <= 18).
        """
        pts = np.asarray(problem["points"], dtype=np.float64)
        n = pts.shape[0]

        if n <= 1:
            return {"simplices": [], "convex_hull": []}
        if n == 2:
            return {"simplices": [], "convex_hull": [[0, 1]]}

        # Helper predicates
        def cross2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            return cross2(a, b, c)

        def incircle3(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
            """
            Robust-ish incircle using 3x3 determinant on lifted coordinates:
            Sign > 0 -> d is inside circumcircle of abc (assuming abc is CCW),
                 < 0 -> outside, == 0 -> cocircular.
            Orientation-corrected to match CCW orientation of abc.
            """
            adx = a[0] - d[0]
            ady = a[1] - d[1]
            bdx = b[0] - d[0]
            bdy = b[1] - d[1]
            cdx = c[0] - d[0]
            cdy = c[1] - d[1]

            ad2 = adx * adx + ady * ady
            bd2 = bdx * bdx + bdy * bdy
            cd2 = cdx * cdx + cdy * cdy

            det = (
                adx * (bdy * cd2 - bd2 * cdy)
                - ady * (bdx * cd2 - bd2 * cdx)
                + ad2 * (bdx * cdy - bdy * cdx)
            )
            if orient(a, b, c) < 0.0:
                det = -det
            return det

        def convex_hull_indices() -> List[int]:
            # Monotone chain for small n; returns hull in CCW order without duplication
            idx = np.lexsort((pts[:, 1], pts[:, 0]))
            lower: List[int] = []
            for i in idx:
                while len(lower) >= 2 and cross2(pts[lower[-2]], pts[lower[-1]], pts[i]) <= 0:
                    lower.pop()
                lower.append(i)
            upper: List[int] = []
            for i in idx[::-1]:
                while len(upper) >= 2 and cross2(pts[upper[-2]], pts[upper[-1]], pts[i]) <= 0:
                    upper.pop()
                upper.append(i)
            return lower[:-1] + upper[:-1]

        if n == 3:
            # If non-collinear, it's a single triangle and 3 hull edges
            if orient(pts[0], pts[1], pts[2]) != 0.0:
                return {
                    "simplices": [[0, 1, 2]],
                    "convex_hull": [[0, 1], [1, 2], [2, 0]],
                }
            # Degenerate -> fall back to SciPy (consistent with reference behavior)
            tri = Delaunay(pts)
            return {"simplices": tri.simplices, "convex_hull": tri.convex_hull}

        if n == 4:
            hull = convex_hull_indices()
            m = len(hull)
            if m == 3:
                # One interior point; triangulate fan from interior point
                hull_set = set(hull)
                interior = next(i for i in range(4) if i not in hull_set)
                a, b, c = hull  # cyclic order from monotone chain
                return {
                    "simplices": [[interior, a, b], [interior, b, c], [interior, c, a]],
                    "convex_hull": [[a, b], [b, c], [c, a]],
                }
            elif m == 4:
                # Convex quadrilateral in cyclic order hull[0..3]
                a, b, c, d = hull
                # Choose diagonal via incircle tests; break ties by shorter diagonal
                s1 = incircle3(pts[a], pts[b], pts[c], pts[d])
                s2 = incircle3(pts[b], pts[c], pts[d], pts[a])
                if s1 > 0.0 and s2 <= 0.0:
                    simplices = [[b, c, d], [b, d, a]]  # diagonal bd
                elif s2 > 0.0 and s1 <= 0.0:
                    simplices = [[a, b, c], [a, c, d]]  # diagonal ac
                else:
                    ac_len2 = float(np.sum((pts[a] - pts[c]) ** 2))
                    bd_len2 = float(np.sum((pts[b] - pts[d]) ** 2))
                    if ac_len2 <= bd_len2:
                        simplices = [[a, b, c], [a, c, d]]
                    else:
                        simplices = [[b, c, d], [b, d, a]]
                hull_edges = [[hull[i], hull[(i + 1) % 4]] for i in range(4)]
                return {"simplices": simplices, "convex_hull": hull_edges}
            # Degenerate cases (collinear/duplicates) -> fall back
            tri = Delaunay(pts)
            return {"simplices": tri.simplices, "convex_hull": tri.convex_hull}

        # Vectorized naive Delaunay for small n (avoid Qhull overhead)
        if n <= 18:
            simplices: List[List[int]] = []
            X = pts[:, 0]
            Y = pts[:, 1]
            # Precompute pairwise differences and squared norms:
            DX = X[:, None] - X[None, :]
            DY = Y[:, None] - Y[None, :]
            D2 = DX * DX + DY * DY

            # Preallocate scratch arrays to avoid per-triple allocations
            det = np.empty(n, dtype=np.float64)
            t1 = np.empty(n, dtype=np.float64)
            t2 = np.empty(n, dtype=np.float64)
            t3 = np.empty(n, dtype=np.float64)

            for i in range(n - 2):
                ai = pts[i]
                for j in range(i + 1, n - 1):
                    aj = pts[j]
                    for k in range(j + 1, n):
                        ak = pts[k]
                        o = (aj[0] - ai[0]) * (ak[1] - ai[1]) - (aj[1] - ai[1]) * (ak[0] - ai[0])
                        if o == 0.0:
                            continue  # collinear triple cannot form triangle

                        # Fetch precomputed vectors for incircle test against all other points
                        adx, ady, ad2 = DX[i], DY[i], D2[i]
                        bdx, bdy, bd2 = DX[j], DY[j], D2[j]
                        cdx, cdy, cd2 = DX[k], DY[k], D2[k]

                        # Compute det using in-place temporaries:
                        # t1 = bdy * cd2 - bd2 * cdy
                        np.multiply(bdy, cd2, out=t1)
                        np.multiply(bd2, cdy, out=det)
                        t1 -= det
                        # t2 = bdx * cd2 - bd2 * cdx
                        np.multiply(bdx, cd2, out=t2)
                        np.multiply(bd2, cdx, out=det)
                        t2 -= det
                        # t3 = bdx * cdy - bdy * cdx
                        np.multiply(bdx, cdy, out=t3)
                        np.multiply(bdy, cdx, out=det)
                        t3 -= det
                        # det = adx * t1 - ady * t2 + ad2 * t3
                        np.multiply(adx, t1, out=det)
                        np.multiply(ady, t2, out=t2)
                        det -= t2
                        np.multiply(ad2, t3, out=t3)
                        det += t3

                        if o < 0.0:
                            det *= -1.0
                        # If any other point lies strictly inside circumcircle, skip
                        if np.any(det > 0.0):
                            continue
                        simplices.append([i, j, k])
            # Convex hull via monotone chain
            hull = convex_hull_indices()
            hull_edges = [[hull[t], hull[(t + 1) % len(hull)]] for t in range(len(hull))]
            return {"simplices": simplices, "convex_hull": hull_edges}

        # General case: delegate to SciPy's Qhull (fast and robust)
        tri = Delaunay(pts)
        return {
            "simplices": tri.simplices,
            "convex_hull": tri.convex_hull,
        }