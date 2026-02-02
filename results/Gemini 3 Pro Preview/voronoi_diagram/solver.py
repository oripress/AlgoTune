import numpy as np
from scipy.spatial import Voronoi
try:
    import fast_utils
except ImportError:
    fast_utils = None

class Solver:
    def solve(self, problem, **kwargs):
        points = problem["points"]
        vor = Voronoi(points)
        
        if fast_utils:
            regions = fast_utils.reorder_regions(vor.regions, vor.point_region)
            ridge_vertices = fast_utils.convert_ridge_vertices(vor.ridge_vertices)
        else:
            regions = [vor.regions[i] for i in vor.point_region]
            ridge_vertices = vor.ridge_vertices
        
        return {
            "vertices": vor.vertices,
            "regions": regions,
            "point_region": np.arange(len(points)),
            "ridge_points": vor.ridge_points,
            "ridge_vertices": ridge_vertices
        }