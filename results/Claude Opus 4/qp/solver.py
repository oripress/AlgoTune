from typing import Any
import numpy as np
import osqp
import scipy.sparse as sp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert to numpy arrays
        P = np.asarray(problem["P"], float)
        q = np.asarray(problem["q"], float)
        G = np.asarray(problem["G"], float)
        h = np.asarray(problem["h"], float)
        A = np.asarray(problem["A"], float)
        b = np.asarray(problem["b"], float)
        
        # Make P symmetric
        P = (P + P.T) / 2
        
        # Combine inequality and equality constraints
        # OSQP expects constraints in the form: l <= Ax <= u
        # For Gx <= h: -inf <= Gx <= h
        # For Ax == b: b <= Ax <= b
        
        if A.size > 0:
            A_combined = np.vstack([G, A])
            l_combined = np.hstack([np.full(G.shape[0], -np.inf), b])
            u_combined = np.hstack([h, b])
        else:
            A_combined = G
            l_combined = np.full(G.shape[0], -np.inf)
            u_combined = h
        
        # Convert to sparse matrices for OSQP
        P_sparse = sp.csc_matrix(P)
        A_sparse = sp.csc_matrix(A_combined)
        
        # Create OSQP object
        m = osqp.OSQP()
        m.setup(P=P_sparse, q=q, A=A_sparse, l=l_combined, u=u_combined,
                eps_abs=1e-8, eps_rel=1e-8, verbose=False)
        
        # Solve
        results = m.solve()
        
        if results.info.status != 'solved':
            raise ValueError(f"Solver failed (status = {results.info.status})")
        
        return {"solution": results.x.tolist()}