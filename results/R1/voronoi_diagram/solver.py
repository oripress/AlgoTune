from scipy.spatial import *

class Solver:
    def solve(self, problem, **kwargs):
        vor = Voronoi(problem["points"], qhull_options='Q0 Qbb Qc')
        return {
            "vertices": vor.vertices.tolist(),
            "regions": vor.regions,
            "point_region": vor.point_region.tolist(),
            "ridge_points": vor.ridge_points.tolist(),
            "ridge_vertices": vor.ridge_vertices
        }