import numpy as np
from scipy.linalg import cho_factor, cho_solve

class Solver:
    def solve(self, problem, **kwargs):
        μ = np.asarray(problem["μ"], dtype=float)
        Σ = np.asarray(problem["Σ"], dtype=float)
        γ = float(problem["γ"])
        n = μ.size
        
        if n == 1:
            return {"w": [1.0]}
        
        two_gamma = 2.0 * γ
        H = two_gamma * Σ  # Hessian matrix

        # Active set method for QP with simplex constraints
        # minimize -μ^T w + γ w^T Σ w  s.t. 1^T w = 1, w >= 0
        free = np.ones(n, dtype=bool)
        
        max_iter = 3 * n + 20
        for iteration in range(max_iter):
            idx = np.where(free)[0]
            m = len(idx)
            if m == 0:
                return None
            
            H_sub = H[np.ix_(idx, idx)]
            μ_sub = μ[idx]
            
            # Solve two linear systems: H_sub * [x1, x2] = [μ_sub, 1]
            rhs = np.empty((m, 2))
            rhs[:, 0] = μ_sub
            rhs[:, 1] = 1.0
            
            try:
                if m > 1:
                    cf = cho_factor(H_sub)
                    sol = cho_solve(cf, rhs)
                else:
                    sol = rhs / H_sub[0, 0]
            except np.linalg.LinAlgError:
                try:
                    H_reg = H_sub.copy()
                    H_reg[np.diag_indices(m)] += 1e-10
                    cf = cho_factor(H_reg)
                    sol = cho_solve(cf, rhs)
                except np.linalg.LinAlgError:
                    sol = np.linalg.lstsq(H_sub, rhs, rcond=None)[0]
            
            x1 = sol[:, 0]
            x2 = sol[:, 1]
            s2 = x2.sum()
            
            if abs(s2) < 1e-15:
                return None
            
            lam = (x1.sum() - 1.0) / s2
            w_sub = x1 - lam * x2
            
            if w_sub.min() >= -1e-10:
                # Feasible: build full weight vector
                w = np.zeros(n)
                w[idx] = np.maximum(w_sub, 0.0)
                ws = w.sum()
                if ws > 1e-15:
                    w /= ws
                
                if m == n:
                    return {"w": w.tolist()}
                
                # Check dual feasibility for bound variables
                # ν_i = H @ w_i - μ_i + λ >= 0 for bound vars
                dual = H @ w - μ + lam
                inactive_idx = np.where(~free)[0]
                dual_inactive = dual[inactive_idx]
                
                if dual_inactive.min() >= -1e-8:
                    return {"w": w.tolist()}
                
                # Release most violated constraint
                worst = inactive_idx[dual_inactive.argmin()]
                free[worst] = True
            else:
                # Fix most negative variable at its bound
                free[idx[w_sub.argmin()]] = False
        
        # Fallback to scipy SLSQP
        from scipy.optimize import minimize as sp_minimize
        
        def obj_grad(w):
            Hw = H @ w
            return -μ @ w + 0.5 * w @ Hw, -μ + Hw
        
        w0 = np.ones(n) / n
        result = sp_minimize(obj_grad, w0, jac=True, method='SLSQP',
                           bounds=[(0, None)] * n,
                           constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1.0,
                                      'jac': lambda w: np.ones(n)})
        w = np.maximum(result.x, 0.0)
        s = w.sum()
        if s > 0:
            w /= s
        return {"w": w.tolist()}