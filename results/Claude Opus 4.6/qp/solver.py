import numpy as np
import scipy.sparse as spa
import osqp

class Solver:
    def solve(self, problem, **kwargs):
        P = np.asarray(problem["P"], dtype=np.float64)
        q = np.asarray(problem["q"], dtype=np.float64)
        G = np.asarray(problem["G"], dtype=np.float64)
        h = np.asarray(problem["h"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        
        n = P.shape[0]
        P = (P + P.T) / 2.0
        
        m_ineq = G.shape[0]
        m_eq = A.shape[0]
        
        if m_ineq > 0 and m_eq > 0:
            A_comb = spa.vstack([spa.csc_matrix(G), spa.csc_matrix(A)], format='csc')
            l = np.concatenate([np.full(m_ineq, -np.inf), b])
            u = np.concatenate([h, b])
        elif m_ineq > 0:
            A_comb = spa.csc_matrix(G)
            l = np.full(m_ineq, -np.inf)
            u = h
        elif m_eq > 0:
            A_comb = spa.csc_matrix(A)
            l = b
            u = b
        else:
            A_comb = spa.csc_matrix((1, n))
            l = np.array([-np.inf])
            u = np.array([np.inf])
        
        P_upper = spa.triu(P, format='csc')
        
        m = osqp.OSQP()
        m.setup(P_upper, q, A_comb, l, u,
                eps_abs=1e-7, eps_rel=1e-7,
                max_iter=4000,
                verbose=False,
                polish=True,
                adaptive_rho=True)
        
        res = m.solve()
        
        if res.info.status not in ('solved', 'solved_inaccurate'):
            raise ValueError(f"Solver failed (status = {res.info.status})")
        
        return {"solution": res.x.tolist(), "objective": float(res.info.obj_val)}