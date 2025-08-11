import numpy as np
from numba import njit

@njit
def _count_cc(n, edges):
    parents = np.arange(n, dtype=np.int32)
    sizes = np.ones(n, dtype=np.int32)
    count = n
    m = edges.shape[0]
    for i in range(m):
        if count == 1:
            break
        u = edges[i, 0]
        v = edges[i, 1]
        # find root of u
        ru = u
        while parents[ru] != ru:
            ru = parents[ru]
        # find root of v
        rv = v
        while parents[rv] != rv:
            rv = parents[rv]
        if ru != rv:
            if sizes[ru] < sizes[rv]:
                parents[ru] = rv
                sizes[rv] += sizes[ru]
            else:
                parents[rv] = ru
                sizes[ru] += sizes[rv]
            count -= 1
    return count

# Warm up JIT compilation to avoid overhead during solve
_count_cc(0, np.empty((0, 2), dtype=np.int32))

class Solver:
    def solve(self, problem, **kwargs):
        n = problem.get("num_nodes", 0)
        if n <= 0:
            return {"number_connected_components": 0}
        edges = problem.get("edges", [])
        m = len(edges)
        if m == 0:
            return {"number_connected_components": int(n)}
        e = np.array(edges, dtype=np.int32)
        cc_count = _count_cc(n, e)
        return {"number_connected_components": int(cc_count)}