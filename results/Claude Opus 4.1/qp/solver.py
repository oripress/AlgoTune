import numpy as np
import osqp
import scipy.sparse as sp
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        # Convert inputs to numpy arrays
        P = np.asarray(problem["P"], dtype=np.float64)
        q = np.asarray(problem["q"], dtype=np.float64)
        G = np.asarray(problem["G"], dtype=np.float64)
        h = np.asarray(problem["h"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        
        n = P.shape[0]
        m = G.shape[0] if G.size > 0 else 0
        p = A.shape[0] if A.size > 0 else 0
        
        # Make P symmetric
        P = (P + P.T) / 2
        
        # Combine inequality and equality constraints
        # OSQP expects Ax = b and l <= Cx <= u
        # We have Gx <= h and Ax = b
        # So we combine them as:
        # [A] x = [b] (lower = upper = b)
        # [G] x <= [h] (lower = -inf, upper = h)
        
        if p > 0 and m > 0:
            A_full = np.vstack([A, G])
            l = np.hstack([b, np.full(m, -np.inf)])
            u = np.hstack([b, h])
        elif p > 0:
            A_full = A
            l = b
            u = b
        elif m > 0:
            A_full = G
            l = np.full(m, -np.inf)
            u = h
        else:
            A_full = np.zeros((1, n))
            l = np.array([-np.inf])
            u = np.array([np.inf])
        
        # Convert to sparse matrices for OSQP
        P_sparse = sp.csc_matrix(P)
        A_sparse = sp.csc_matrix(A_full)
        
        # Setup OSQP problem
        osqp_solver = osqp.OSQP()
        osqp_solver.setup(P=P_sparse, q=q, A=A_sparse, l=l, u=u, 
                         eps_abs=1e-8, eps_rel=1e-8, verbose=False)
        
        # Solve
        results = osqp_solver.solve()
        
        if results.info.status != 'solved':
            raise ValueError(f"Solver failed (status = {results.info.status})")
        
        # Compute objective value
        x = results.x
        obj_value = 0.5 * x @ P @ x + q @ x
        
        return {"solution": x.tolist(), "objective": float(obj_value)}