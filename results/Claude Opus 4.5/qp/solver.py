import numpy as np
import osqp
from scipy import sparse
from scipy.optimize import minimize

class Solver:
    def solve(self, problem, **kwargs):
        P = np.asarray(problem["P"], dtype=np.float64)
        q = np.asarray(problem["q"], dtype=np.float64).ravel()
        G = np.asarray(problem["G"], dtype=np.float64)
        h = np.asarray(problem["h"], dtype=np.float64).ravel()
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64).ravel()
        
        n = P.shape[0]
        m = G.shape[0]
        p = A.shape[0]
        
        # Make P symmetric
        P_sym = (P + P.T) * 0.5
        
        # For small problems, try direct scipy solver
        if n <= 20 and m <= 50 and p <= 20:
            return self._solve_scipy(P_sym, q, G, h, A, b, n, m, p)
        else:
            return self._solve_osqp(P_sym, q, G, h, A, b, n, m, p)
    
    def _solve_scipy(self, P, q, G, h, A, b, n, m, p):
        def objective(x):
            return 0.5 * x @ P @ x + q @ x
        
        def grad(x):
            return P @ x + q
        
        constraints = []
        if m > 0:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: h - G @ x,
                'jac': lambda x: -G
            })
        if p > 0:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: A @ x - b,
                'jac': lambda x: A
            })
        
        x0 = np.zeros(n)
        result = minimize(objective, x0, jac=grad, method='SLSQP', 
                         constraints=constraints, options={'ftol': 1e-9, 'maxiter': 1000})
        
        if not result.success:
            # Fall back to OSQP
            return self._solve_osqp(P, q, G, h, A, b, n, m, p)
        
        obj_val = 0.5 * result.x @ P @ result.x + q @ result.x
        return {"solution": result.x.tolist(), "objective": float(obj_val)}
    
    def _solve_osqp(self, P, q, G, h, A, b, n, m, p):
        P_sparse = sparse.triu(P, format='csc')
        
        if p > 0:
            A_combined = sparse.csc_matrix(np.vstack([G, A]))
            l = np.concatenate([np.full(m, -np.inf), b])
            u = np.concatenate([h, b])
        else:
            A_combined = sparse.csc_matrix(G)
            l = np.full(m, -np.inf)
            u = h
        
        solver = osqp.OSQP()
        solver.setup(
            P_sparse, q, A_combined, l, u,
            eps_abs=1e-8,
            eps_rel=1e-8,
            verbose=False,
        )
        
        result = solver.solve()
        
        if result.info.status not in ['solved', 'solved_inaccurate']:
            raise ValueError(f"Solver failed (status = {result.info.status})")
        
        return {"solution": result.x.tolist(), "objective": float(result.info.obj_val)}