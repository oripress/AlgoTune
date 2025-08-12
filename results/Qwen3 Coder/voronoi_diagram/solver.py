import math
import random
from typing import Any
import numpy as np
from scipy.spatial import qhull
class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the Voronoi diagram construction problem using scipy.spatial.

        :param problem: A dictionary representing the Voronoi problem.
        :return: A dictionary with keys:
                 "vertices": List of coordinates of the Voronoi vertices.
                 "regions": List of lists, where each list contains the indices of the Voronoi vertices
                           forming a region.
                 "point_region": List mapping each input point to its corresponding region.
                 "ridge_points": List of pairs of input points, whose Voronoi regions share an edge.
                 "ridge_vertices": List of pairs of indices of Voronoi vertices forming a ridge.
        """
        points = np.array(problem["points"])
        
        # Use Qhull to compute Voronoi diagram
        vor = qhull.Voronoi(points)
        
        solution = {
            "vertices": vor.vertices,
            "regions": vor.regions,
            "point_region": vor.point_region,
            "ridge_points": vor.ridge_points,
            "ridge_vertices": vor.ridge_vertices,
        }
        
        return solution