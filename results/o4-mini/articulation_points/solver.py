import numpy as np
from solver_numba import solve_nb

class Solver:
    def solve(self, problem, **kwargs):
        n = problem["num_nodes"]
        edges = problem["edges"]
        m = len(edges)
        # trivial empty cases
        if n <= 0 or m == 0:
            return {"articulation_points": []}
        # build edge arrays
        u_arr = np.empty(m, dtype=np.int32)
        v_arr = np.empty(m, dtype=np.int32)
        for i, (u, v) in enumerate(edges):
            u_arr[i] = u
            v_arr[i] = v
        # call Numba‐compiled Tarjan solver
        ap_mask = solve_nb(n, u_arr, v_arr)
        # extract indices of articulation points
        aps = np.nonzero(ap_mask)[0]
        return {"articulation_points": aps.tolist()}