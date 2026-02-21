import numpy as np
import osqp
from scipy import sparse
from typing import Any

class Solver:
    _cache = {}

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[float]] | None:
        mu = np.array(problem["μ"], dtype=float, copy=False)
        Sigma = np.array(problem["Σ"], dtype=float, copy=False)
        gamma = float(problem["γ"])
        n = mu.size
        
        if n not in Solver._cache:
            P_indptr = np.zeros(n + 1, dtype=np.int32)
            P_indptr[1:] = np.cumsum(np.arange(1, n + 1))
            
            # Column-major upper triangular indices
            tril_idx = np.tril_indices(n)
            
            A_val = np.ones(2 * n, dtype=float)
            A_indices = np.empty(2 * n, dtype=np.int32)
            A_indices[0::2] = 0
            A_indices[1::2] = np.arange(1, n + 1, dtype=np.int32)
            A_indptr = np.arange(0, 2 * n + 1, 2, dtype=np.int32)
            A = sparse.csc_matrix((A_val, A_indices, A_indptr), shape=(n + 1, n))
            
            l = np.zeros(n + 1, dtype=float)
            l[0] = 1.0
            u = np.full(n + 1, np.inf, dtype=float)
            u[0] = 1.0
            
            prob = osqp.OSQP()
            
            P_data = 2 * gamma * Sigma.T[tril_idx]
            P_indices = tril_idx[1]
            P = sparse.csc_matrix((P_data, P_indices, P_indptr), shape=(n, n))
            
            q = -mu
            
            prob.setup(P, q, A, l, u, verbose=False, eps_abs=1e-7, eps_rel=1e-7)
            
            Solver._cache[n] = {
                'prob': prob,
                'tril_idx': tril_idx
            }
        else:
            cache = Solver._cache[n]
            prob = cache['prob']
            tril_idx = cache['tril_idx']
            
            P_data = 2 * gamma * Sigma.T[tril_idx]
            q = -mu
            prob.update(Px=P_data, q=q)
            
        res = prob.solve()
        
        if res.info.status_val in (1, 2):
            return {"w": res.x.tolist()}
        return None