import numpy as np
from typing import Any
import osqp
import scipy.sparse as sp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Extract and convert input data to numpy arrays - optimized version
        P = np.array(problem["P"], dtype=np.float64, order='C')
        q = np.array(problem["q"], dtype=np.float64, order='C')
        G = np.array(problem["G"], dtype=np.float64, order='C')
        h = np.array(problem["h"], dtype=np.float64, order='C')
        A = np.array(problem["A"], dtype=np.float64, order='C')
        b = np.array(problem["b"], dtype=np.float64, order='C')
        n = P.shape[0]
        
        # Ensure P is symmetric
        P = (P + P.T) / 2
        
        # Convert to sparse matrices for OSQP
        P_sparse = sp.csc_matrix(P, dtype=np.float64)
        A_sparse = sp.csc_matrix(np.vstack([G, A]), dtype=np.float64)
        l = np.concatenate([-np.inf * np.ones(len(h), dtype=np.float64), b])
        u = np.concatenate([h, b])
        
        # Create OSQP problem with optimized settings
        model = osqp.OSQP()
        model.setup(P=P_sparse, q=q, A=A_sparse, l=l, u=u, 
                   eps_abs=1e-6, eps_rel=1e-6, verbose=False, polish=True, max_iter=500)
        result = model.solve()
        
        # Check solver status
        if result.info.status not in ('solved', 'solved inaccurate'):
            raise ValueError(f"Solver failed (status = {result.info.status})")
        
        return {"solution": result.x.tolist(), "objective": float(result.info.obj_val)}