import numpy as np
from scipy.linalg import cho_factor, cho_solve

class Solver:
    def solve(self, problem, **kwargs):
        c = np.array(problem["c"], dtype=np.float64)
        A = np.array(problem["A"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        
        m, n = A.shape
        
        # Initial point
        x = np.ones(n, dtype=np.float64)
        nu = np.zeros(m, dtype=np.float64)
        
        # Parameters
        MAX_ITERS = 100
        TOL = 1e-8
        ALPHA = 0.01
        BETA = 0.5
        
        for i in range(MAX_ITERS):
            # Residuals
            inv_x = 1.0 / x
            ATnu = A.T @ nu
            r_dual = c - inv_x + ATnu
            Ax = A @ x
            r_prim = Ax - b
            
            norm_r_dual = np.linalg.norm(r_dual)
            norm_r_prim = np.linalg.norm(r_prim)
            residual_norm = np.sqrt(norm_r_dual**2 + norm_r_prim**2)
            
            if residual_norm < TOL:
                break
            
            # Newton step
            # M = (A * x) @ (A * x).T
            B = A * x
            M = B @ B.T
            
            x2 = x**2
            # rhs = r_prim - A @ (x^2 * r_dual)
            rhs = r_prim - A @ (x2 * r_dual)
            
            if m > 0:
                try:
                    c_and_lower = cho_factor(M)
                    d_nu = cho_solve(c_and_lower, rhs)
                except np.linalg.LinAlgError:
                    d_nu = np.linalg.lstsq(M, rhs, rcond=None)[0]
            else:
                d_nu = np.zeros(0)
            
            ATdnu = A.T @ d_nu
            d_x = -x2 * (r_dual + ATdnu)
            
            # Line search
            neg_idx = d_x < 0
            s = 1.0
            if np.any(neg_idx):
                s_max = np.min(-x[neg_idx] / d_x[neg_idx])
                s = min(1.0, 0.99 * s_max)
            
            current_res_norm = residual_norm
            
            # Precompute for line search
            Adx = A @ d_x
            
            for _ in range(20):
                x_new = x + s * d_x
                
                r_prim_new = r_prim + s * Adx
                
                inv_x_new = 1.0 / x_new
                r_dual_new = c - inv_x_new + ATnu + s * ATdnu
                
                new_res_norm = np.sqrt(np.linalg.norm(r_dual_new)**2 + np.linalg.norm(r_prim_new)**2)
                
                if new_res_norm <= (1 - ALPHA * s) * current_res_norm:
                    x = x_new
                    nu = nu + s * d_nu
                    break
                
                s *= BETA
            else:
                x = x + s * d_x
                nu = nu + s * d_nu
        
        return {"solution": x.tolist()}