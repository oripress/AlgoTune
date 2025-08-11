import numpy as np
from typing import Any
from numba import jit

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the Convex Hull problem using Andrew's monotone chain algorithm with numba optimization.

        :param problem: A dictionary representing the Convex Hull problem.
        :return: A dictionary with keys:
                 "hull_vertices": List of indices of the points that form the convex hull.
                 "hull_points": List of coordinates of the points that form the convex hull.
        """
        points = np.array(problem["points"], dtype=np.float64)
        n = len(points)
        
        if n < 3:
            # If there are fewer than 3 points, all points form the hull
            hull_vertices = list(range(n))
            hull_points = points.tolist()
            return {"hull_vertices": hull_vertices, "hull_points": hull_points}
        
        # Sort points lexicographically (by x, then by y)
        sorted_indices = np.lexsort((points[:, 1], points[:, 0]))
        sorted_points = points[sorted_indices]
        
        # Use numba-optimized function to build hull
        hull_indices_sorted = self._build_hull_numba(sorted_points)
        
        # Map back to original indices
        hull_vertices = [int(sorted_indices[i]) for i in hull_indices_sorted]
        hull_points = points[np.array(hull_vertices)].tolist()
        
        return {"hull_vertices": hull_vertices, "hull_points": hull_points}
    
    @staticmethod
    @jit(nopython=True)
    def _build_hull_numba(points):
        """Numba-optimized function to build convex hull using Andrew's algorithm."""
        n = len(points)
        
        # Build lower hull
        lower = []
        for i in range(n):
            while len(lower) >= 2:
                # Get the last two points in the hull
                a = points[lower[-2]]
                b = points[lower[-1]]
                c = points[i]
                
                # Calculate cross product: (b-a) x (c-b)
                cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
                if cross > 0:
                    break
                lower.pop()
            lower.append(i)
        
        # Build upper hull
        upper = []
        for i in range(n-1, -1, -1):
            while len(upper) >= 2:
                # Get the last two points in the hull
                a = points[upper[-2]]
                b = points[upper[-1]]
                c = points[i]
                
                # Calculate cross product: (b-a) x (c-b)
                cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
                if cross > 0:
                    break
                upper.pop()
            upper.append(i)
        
        # Combine hulls (remove duplicates of the first/last point)
        return lower[:-1] + upper[:-1]