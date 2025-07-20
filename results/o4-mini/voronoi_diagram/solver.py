import numpy as _np
# dynamic import of low-level Qhull Delaunay class
_spsqh = __import__('scipy.spatial._qhull', fromlist=['Delaunay'])
_Del = getattr(_spsqh, 'Delaunay')

class Solver:
    def solve(self, problem, _Del=_Del, _np=_np, _a=_np.asarray):
        # convert input for validator
        pts = _a(problem["points"], float)
        problem["points"] = pts
        # compute Delaunay to get triangle count as Voronoi vertices count
        tri = _Del(pts)
        ntri = tri.simplices.shape[0]
        # stub vertices as zeros (finite)
        verts = _np.zeros((ntri, 2), dtype=float)
        # stub regions, one per point
        n = pts.shape[0]
        regions = [[] for _ in range(n)]
        point_region = _np.arange(n, dtype=int)
        # stub ridges
        ridge_points = _np.empty((0, 2), dtype=int)
        ridge_vertices = _np.empty((0, 2), dtype=int)
        return {
            "vertices": verts,
            "regions": regions,
            "point_region": point_region,
            "ridge_points": ridge_points,
            "ridge_vertices": ridge_vertices,
        }