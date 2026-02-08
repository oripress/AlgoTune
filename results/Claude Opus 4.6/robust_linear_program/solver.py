from typing import Any
import numpy as np
from scipy import sparse
import ecos

class Solver:
    def solve(self, problem, **kwargs):
        c = np.asarray(problem["c"], dtype=np.float64)
        b_vals = np.asarray(problem["b"], dtype=np.float64)
        n = len(c)
        m = len(b_vals)
        cone_q = []
        G_blocks = []
        h_blocks = []
        for i in range(m):
            Pi = np.asarray(problem["P"][i], dtype=np.float64)
            qi = np.asarray(problem["q"][i], dtype=np.float64)
            if Pi.ndim < 2:
                Pi = Pi.reshape(n, 1)
            k = Pi.shape[1]
            cone_q.append(1 + k)
            blk = np.empty((1 + k, n), dtype=np.float64)
            blk[0] = qi
            blk[1:] = -Pi.T
            G_blocks.append(blk)
            hb = np.zeros(1 + k)
            hb[0] = b_vals[i]
            h_blocks.append(hb)
        G = np.vstack(G_blocks)
        h = np.concatenate(h_blocks)
        G_sp = sparse.csc_matrix(G)
        sol = ecos.solve(c, G_sp, h, {'l': 0, 'q': cone_q}, verbose=False)
        flag = sol['info']['exitFlag']
        if flag == 0 or flag == 10:
            x = sol['x']
            return {"objective_value": float(c @ x), "x": x}
        return {"objective_value": float("inf"), "x": np.full(n, np.nan)}