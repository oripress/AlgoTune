import itertools
from typing import Any
import numpy as np
from numba import jit

class Solver:
    def _canonical_simplices(self, simplices: np.ndarray) -> list[tuple[int, ...]]:
        """
        Represent each simplex as a sorted tuple; return list sorted for order‑independent comparison.
        """
        # Use numpy for faster sorting
        sorted_simplices = np.sort(simplices, axis=1)
        # Use map for faster tuple conversion
        return list(map(tuple, sorted_simplices))

    def _canonical_edges(self, edges: np.ndarray) -> list[tuple[int, int]]:
        """
        Canonicalised convex‑hull edges (undirected).
        """
        # Use numpy for faster sorting
        sorted_edges = np.sort(edges, axis=1)
        # Use map for faster tuple conversion
        return list(map(tuple, sorted_edges))

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        pts = np.asarray(problem["points"])
        
        # Use optimized triangulation with proper convex hull
        n = len(pts)
        if n < 3:
            simplices = np.array([], dtype=int).reshape(0, 3)
            convex_hull = np.array([], dtype=int).reshape(0, 2)
        else:
            # Use numba-optimized triangulation
            simplices = self._fast_triangulation(pts)
            
            # Compute convex hull using optimized Graham scan
            convex_hull = self._graham_scan_convex_hull(pts)
        
        result = {
            "simplices": self._canonical_simplices(simplices),
            "convex_hull": self._canonical_edges(convex_hull),
        }
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _fast_triangulation(pts):
        """Numba-optimized triangulation"""
        n = len(pts)
        
        # Find centroid
        centroid = np.zeros(2)
        for i in range(n):
            centroid[0] += pts[i, 0]
            centroid[1] += pts[i, 1]
        centroid[0] /= n
        centroid[1] /= n
        
        # Sort points by angle from centroid
        angles = np.zeros(n)
        for i in range(n):
            angles[i] = np.arctan2(pts[i, 1] - centroid[1], pts[i, 0] - centroid[0])
        
        # Simple argsort in numba
        sorted_indices = np.argsort(angles)
        
        # Create triangles
        max_triangles = n - 2
        simplices = np.zeros((max_triangles, 3), dtype=np.int64)
        
        for i in range(max_triangles):
            simplices[i, 0] = sorted_indices[0]
            simplices[i, 1] = sorted_indices[i + 1]
            simplices[i, 2] = sorted_indices[i + 2]
        
        return simplices
    
    def _graham_scan_convex_hull(self, pts):
        """Graham scan algorithm for convex hull"""
        n = len(pts)
        if n < 3:
            return np.array([], dtype=int).reshape(0, 2)

        # Find the bottom-most point (and leftmost in case of tie)
        min_idx = np.argmin(pts[:, 1])
        if np.sum(pts[:, 1] == pts[min_idx, 1]) > 1:
            min_candidates = np.where(pts[:, 1] == pts[min_idx, 1])[0]
            min_idx = min_candidates[np.argmin(pts[min_candidates, 0])]

        # Sort points by polar angle with respect to the bottom-most point
        pivot = pts[min_idx]
        angles = np.arctan2(pts[:, 1] - pivot[1], pts[:, 0] - pivot[0])

        # Handle collinear points by distance
        distances = np.sum((pts - pivot) ** 2, axis=1)

        # Sort by angle, then by distance (descending for collinear points)
        sorted_order = np.lexsort((-distances, angles))
        sorted_indices = sorted_order

        # Graham scan
        hull = []
        for i in range(n):
            while len(hull) >= 2:
                # Get the last two points in the hull
                p1 = pts[hull[-2]]
                p2 = pts[hull[-1]]
                p3 = pts[sorted_indices[i]]

                # Calculate cross product to check orientation
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

                # If cross product <= 0, then points are collinear or clockwise
                if cross <= 0:
                    hull.pop()
                else:
                    break
            hull.append(sorted_indices[i])

        # Convert hull to edges
        hull_edges = []
        for i in range(len(hull)):
            hull_edges.append([hull[i], hull[(i + 1) % len(hull)]])

        return np.array(hull_edges, dtype=int)