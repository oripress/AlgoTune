from typing import Any
import numpy as np
import osqp
from scipy import sparse
class Solver:
    def __init__(self):
        self.prob = osqp.OSQP()

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        P = np.asarray(problem["P"], float)
        q = np.asarray(problem["q"], float)
        G = np.asarray(problem["G"], float)
        h = np.asarray(problem["h"], float)
        A = np.asarray(problem["A"], float)
        b = np.asarray(problem["b"], float)
        
        # Symmetrize P and take upper triangular part
        P = (P + P.T) / 2.0
        P_sparse = sparse.csc_matrix(np.triu(P))
        
        # Constraints: 
        # Gx <= h
        # Ax == b
        # We can stack them:
        # [G; A] x <= [h; b]
        # [G; A] x >= [-inf; b]
        
        A_osqp = sparse.csc_matrix(np.vstack((G, A)))
        
        m = h.shape[0]
        l = np.concatenate((np.full(m, -np.inf), b))
        u = np.concatenate((h, b))
        
        self.prob.setup(P_sparse, q, A_osqp, l, u, verbose=False, eps_abs=1e-8, eps_rel=1e-8, adaptive_rho=False, polish=False)
        res = self.prob.solve()
        
        return {"solution": res.x.tolist(), "objective": res.info.obj_val}