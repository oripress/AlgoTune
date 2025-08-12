import numpy as np
import numba

@numba.njit(cache=True)
def _kruskal(n, us, vs, ws, idx):
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int64)
    max_edges = n - 1 if n > 0 else 0
    out_u = np.empty(max_edges, dtype=np.int64)
    out_v = np.empty(max_edges, dtype=np.int64)
    out_w = np.empty(max_edges, dtype=np.float64)
    count = 0
    m = idx.shape[0]
    for k in range(m):
        j = idx[k]
        u = us[j]
        v = vs[j]
        # find root of u
        ru = u
        while parent[ru] != ru:
            parent[ru] = parent[parent[ru]]
            ru = parent[ru]
        # find root of v
        rv = v
        while parent[rv] != rv:
            parent[rv] = parent[parent[rv]]
            rv = parent[rv]
        if ru == rv:
            continue
        # union by rank
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1
        # normalize ordering u < v
        if u > v:
            tmp = u
            u = v
            v = tmp
        out_u[count] = u
        out_v[count] = v
        out_w[count] = ws[j]
        count += 1
        if count == max_edges:
            break
    return out_u[:count], out_v[:count], out_w[:count]

class Solver:
    def __init__(self):
        # Warm up Numba to avoid compile overhead in solve
        us = np.array([0, 1], dtype=np.int64)
        vs = np.array([1, 0], dtype=np.int64)
        ws = np.array([0.0, 0.0], dtype=np.float64)
        idx = np.array([0, 1], dtype=np.int64)
        _kruskal(2, us, vs, ws, idx)

    def solve(self, problem, **kwargs):
        n = int(problem.get("num_nodes", 0))
        edges = problem.get("edges", [])
        if n <= 0 or not edges:
            return {"mst_edges": []}
        # Pack edges into numpy arrays using fromiter
        m = len(edges)
        us = np.fromiter((e[0] for e in edges), dtype=np.int64, count=m)
        vs = np.fromiter((e[1] for e in edges), dtype=np.int64, count=m)
        ws = np.fromiter((e[2] for e in edges), dtype=np.float64, count=m)
        # Stable argsort by weight to preserve input order on ties
        idx = np.argsort(ws, kind='mergesort').astype(np.int64)
        # Compute MST via Kruskal in Numba
        ures, vres, wres = _kruskal(n, us, vs, ws, idx)
        k = ures.shape[0]
        if k > 1:
            # sort MST edges by (u, v) using numpy lexsort
            order = np.lexsort((vres, ures))
            ures = ures[order]
            vres = vres[order]
            wres = wres[order]
        # Build result list
        mst_edges = [[int(ures[i]), int(vres[i]), float(wres[i])] for i in range(k)]
        return {"mst_edges": mst_edges}