import numpy as np
import ecos
import scipy.sparse as sp
from typing import Any

class Solver:
    def solve(self, problem: dict[str, np.ndarray], **kwargs) -> dict[str, Any]:
        c = np.asarray(problem["c"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        P = np.asarray(problem["P"], dtype=np.float64)
        q = np.asarray(problem["q"], dtype=np.float64)
        
        m = len(P)
        n = len(c)
        
        if m == 0:
            # Unconstrained problem
            # If c is not all zeros, it's unbounded.
            # But let's just let ECOS handle it or handle it manually.
            pass
        if m > 0:
            G = np.empty((m, n + 1, n), dtype=np.float64)
            G[:, 0, :] = q
            G[:, 1:, :] = -P.transpose(0, 2, 1)
            G = G.reshape(m * (n + 1), n)
            
            h = np.zeros(m * (n + 1), dtype=np.float64)
            h[0::n+1] = b
            
            q_dims = [n + 1] * m
        else:
            G = np.zeros((0, n))
            h = np.zeros(0)
            q_dims = []
            
        if m > 0:
            data = G.ravel(order='F')
            indices = np.empty((n, m * (n + 1)), dtype=np.int32)
            indices[:] = np.arange(m * (n + 1), dtype=np.int32)
            indices = indices.ravel()
            indptr = np.arange(0, n * m * (n + 1) + 1, m * (n + 1), dtype=np.int32)
            G_sparse = sp.csc_matrix.__new__(sp.csc_matrix)
            G_sparse.data = data
            G_sparse.indices = indices
            G_sparse.indptr = indptr
            G_sparse._shape = (m * (n + 1), n)
        else:
            G_sparse = sp.csc_matrix((0, n))
        dims = {'l': 0, 'q': q_dims}
        
        try:
            solution = ecos.solve(c, G_sparse, h, dims, verbose=False)
            
            if solution['info']['exitFlag'] in (0, 10):
                x = np.array(solution['x'])
                obj = solution['info']['pcost']
                return {"objective_value": obj, "x": x}
            else:
                return {"objective_value": float("inf"), "x": np.full(n, np.nan)}
        except Exception:
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}