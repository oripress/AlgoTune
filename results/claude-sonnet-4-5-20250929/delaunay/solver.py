import numpy as np
from typing import Any
import numba as nb
from numba.typed import List

@nb.njit(cache=True, fastmath=True)
def cross_product(o, a, b):
    """2D cross product."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

@nb.njit(cache=True, fastmath=True)
def in_circumcircle(p, a, b, c):
    """Check if point p is inside circumcircle of triangle abc."""
    ax, ay = a[0] - p[0], a[1] - p[1]
    bx, by = b[0] - p[0], b[1] - p[1]
    cx, cy = c[0] - p[0], c[1] - p[1]
    
    det = (ax * ax + ay * ay) * (bx * cy - cx * by) - \
          (bx * bx + by * by) * (ax * cy - cx * ay) + \
          (cx * cx + cy * cy) * (ax * by - bx * ay)
    return det > 1e-10

@nb.njit(cache=True, fastmath=True)
def delaunay_bowyer_watson(points):
    """Bowyer-Watson algorithm for Delaunay triangulation."""
    n = len(points)
    
    # Find bounding super-triangle
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    for p in points:
        min_x = min(min_x, p[0])
        max_x = max(max_x, p[0])
        min_y = min(min_y, p[1])
        max_y = max(max_y, p[1])
    
    dx = max_x - min_x
    dy = max_y - min_y
    dmax = max(dx, dy)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    
    # Super-triangle vertices
    p1 = np.array([mid_x - 20 * dmax, mid_y - dmax])
    p2 = np.array([mid_x, mid_y + 20 * dmax])
    p3 = np.array([mid_x + 20 * dmax, mid_y - dmax])
    
    # Initialize with super-triangle
    triangles = List()
    triangles.append((n, n+1, n+2))
    
    all_points = np.vstack((points, p1.reshape(1, -1), p2.reshape(1, -1), p3.reshape(1, -1)))
    
    # Add points incrementally
    for i in range(n):
        bad_triangles = List()
        
        # Find triangles whose circumcircle contains the point
        for j in range(len(triangles)):
            tri = triangles[j]
            if in_circumcircle(all_points[i], all_points[tri[0]], all_points[tri[1]], all_points[tri[2]]):
                bad_triangles.append(j)
        
        # Find boundary of polygonal hole
        polygon = List()
        for j in bad_triangles:
            tri = triangles[j]
            edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for edge in edges:
                shared = False
                for k in bad_triangles:
                    if k == j:
                        continue
                    other_tri = triangles[k]
                    other_edges = [(other_tri[0], other_tri[1]), (other_tri[1], other_tri[2]), (other_tri[2], other_tri[0])]
                    for other_edge in other_edges:
                        if (edge[0] == other_edge[0] and edge[1] == other_edge[1]) or \
                           (edge[0] == other_edge[1] and edge[1] == other_edge[0]):
                            shared = True
                            break
                    if shared:
                        break
                if not shared:
                    polygon.append(edge)
        
        # Remove bad triangles
        new_triangles = List()
        for j in range(len(triangles)):
            if j not in bad_triangles:
                new_triangles.append(triangles[j])
        triangles = new_triangles
        
        # Re-triangulate the polygonal hole
        for edge in polygon:
            triangles.append((edge[0], edge[1], i))
    
    # Remove triangles that contain super-triangle vertices
    final_triangles = List()
    for tri in triangles:
        if tri[0] < n and tri[1] < n and tri[2] < n:
            final_triangles.append(tri)
    
    return final_triangles

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        pts = problem["points"]
        if not isinstance(pts, np.ndarray):
            pts = np.asarray(pts, dtype=np.float64)
        
        # Use scipy for now as Numba implementation needs more work
        from scipy.spatial import Delaunay as SciPyDelaunay
        tri = SciPyDelaunay(pts)
        
        # Optimize canonicalization
        simplices = np.sort(tri.simplices, axis=1)
        key = simplices[:, 0] * 1000000000000 + simplices[:, 1] * 1000000 + simplices[:, 2]
        idx = np.argsort(key, kind='quicksort')
        canonical_simplices = simplices[idx]
        
        convex_hull = np.sort(tri.convex_hull, axis=1)
        key = convex_hull[:, 0] * 1000000 + convex_hull[:, 1]
        idx = np.argsort(key, kind='quicksort')
        canonical_edges = convex_hull[idx]
        
        return {
            "simplices": canonical_simplices.tolist(),
            "convex_hull": canonical_edges.tolist(),
        }