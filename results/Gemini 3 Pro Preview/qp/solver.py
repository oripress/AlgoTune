from typing import Any
import numpy as np
try:
    import osqp
except ImportError:
    osqp = None
import cvxpy as cp
try:
    from fast_utils import prepare_data
except ImportError:
    prepare_data = None

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        # Check for special cases: Unconstrained or Equality only
        # We need to inspect G and A sizes.
        # Accessing problem dict is fast.
        G_in = problem["G"]
        h_in = problem["h"]
        A_in = problem["A"]
        b_in = problem["b"]
        
        m = len(h_in)
        p = len(b_in)
        
        if m == 0:
            # No inequality constraints
            P_in = problem.get("P", problem.get("Q"))
            q_in = problem["q"]
            P = np.array(P_in, dtype=float)
            q = np.array(q_in, dtype=float)
            n = len(q)
            
            # Symmetrize P
            P = (P + P.T) * 0.5
            
            if p == 0:
                # Unconstrained: Solve P x = -q
                # Use Cholesky if possible, else solve
                try:
                    L = np.linalg.cholesky(P)
                    y = np.linalg.solve(L, -q)
                    x = np.linalg.solve(L.T, y)
                    return {"solution": x.tolist()}
                except np.linalg.LinAlgError:
                    # P might be singular or not PD (just PSD)
                    # Fallback to lstsq or pinv?
                    # Or just fallback to OSQP
                    pass
            else:
                # Equality constrained only
                # Solve KKT system
                # [ P  A^T ] [ x ] = [ -q ]
                # [ A   0  ] [ v ]   [  b ]
                A = np.array(A_in, dtype=float)
                if A.ndim == 1: A = A.reshape(1, -1)
                b = np.array(b_in, dtype=float)
                
                KKT_mat = np.block([[P, A.T], [A, np.zeros((p, p))]])
                rhs = np.concatenate([-q, b])
                
                try:
                    sol = np.linalg.solve(KKT_mat, rhs)
                    x = sol[:n]
                    return {"solution": x.tolist()}
                except np.linalg.LinAlgError:
                    pass

        if osqp is not None and prepare_data is not None:
            return self.solve_osqp_fast(problem)
        elif osqp is not None:
            return self.solve_osqp(problem)
        
        # Fallback to CVXPY
        P_key = "P" if "P" in problem else "Q"
        P = np.asarray(problem[P_key], float)
        q = np.asarray(problem["q"], float)
        G = np.asarray(problem["G"], float)
        h = np.asarray(problem["h"], float)
        A = np.asarray(problem["A"], float)
        b = np.asarray(problem["b"], float)
        n = P.shape[0]

        P = (P + P.T) / 2
        x = cp.Variable(n)
        objective = 0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x
        constraints = [G @ x <= h, A @ x == b]
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8)
        return {"solution": x.value.tolist()}

    def solve_osqp_fast(self, problem):
        P_sparse, q, A_sparse, l, u = prepare_data(problem)
        
        solver = osqp.OSQP()
        solver.setup(P=P_sparse, q=q, A=A_sparse, l=l, u=u, verbose=False, eps_abs=1e-8, eps_rel=1e-8)
        res = solver.solve()
        
        return {"solution": res.x.tolist()}

    def solve_osqp(self, problem):
        # ... (previous implementation kept for safety, though likely not used if fast_utils works)
        P_in = problem.get("P", problem.get("Q"))
        q = np.array(problem["q"], dtype=float)
        n = q.shape[0]
        
        P = np.array(P_in, dtype=float)
        P = (P + P.T) * 0.5
        
        G_in = problem["G"]
        h_in = problem["h"]
        A_in = problem["A"]
        b_in = problem["b"]
        
        h = np.array(h_in, dtype=float)
        b = np.array(b_in, dtype=float)
        m = h.shape[0]
        p = b.shape[0]
        
        l = np.empty(m + p, dtype=float)
        u = np.empty(m + p, dtype=float)
        
        if m > 0:
            l[:m] = -np.inf
            u[:m] = h
        if p > 0:
            l[m:] = b
            u[m:] = b
            
        A_osqp = np.empty((m + p, n), dtype=float)
        
        if m > 0:
            G = np.array(G_in, dtype=float, copy=False)
            if G.ndim == 1:
                A_osqp[:m] = G.reshape(1, -1)
            else:
                A_osqp[:m] = G
                
        if p > 0:
            A = np.array(A_in, dtype=float, copy=False)
            if A.ndim == 1:
                A_osqp[m:] = A.reshape(1, -1)
            else:
                A_osqp[m:] = A
                
        import scipy.sparse as spa
        P_sparse = spa.triu(P, format='csc')
        A_sparse = spa.csc_matrix(A_osqp)
        
        solver = osqp.OSQP()
        solver.setup(P=P_sparse, q=q, A=A_sparse, l=l, u=u, verbose=False, eps_abs=1e-8, eps_rel=1e-8)
        res = solver.solve()
        
        return {"solution": res.x.tolist()}
        solver = osqp.OSQP()
        solver.setup(P=P_sparse, q=q, A=A_sparse, l=l, u=u, verbose=False, eps_abs=1e-8, eps_rel=1e-8)
        res = solver.solve()
        
        return {"solution": res.x.tolist()}