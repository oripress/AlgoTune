import numpy as np
from scipy.linalg import cho_factor, cho_solve

class Solver:
    def solve(self, problem, **kwargs):
        c = np.array(problem["c"], dtype=np.float64)
        A = np.array(problem["A"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        m, n = A.shape
        
        # Find initial feasible point x > 0 with Ax = b
        # Try minimum norm solution first
        if m <= n:
            AAT = A @ A.T
            try:
                L_AAT = cho_factor(AAT)
                x = A.T @ cho_solve(L_AAT, b)
            except Exception:
                x = np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            x = np.linalg.lstsq(A, b, rcond=None)[0]
        
        if np.min(x) <= 1e-8:
            # Find a feasible strictly positive point using linprog
            from scipy.optimize import linprog
            res = linprog(np.zeros(n), A_eq=A, b_eq=b, bounds=[(1e-6, None)] * n, method='highs')
            if res.success:
                x = res.x.copy()
            else:
                # Fallback to CVXPY
                import cvxpy as cp
                xv = cp.Variable(n)
                prob = cp.Problem(cp.Minimize(c.T @ xv - cp.sum(cp.log(xv))), [A @ xv == b])
                prob.solve(solver="CLARABEL")
                return {"solution": xv.value.tolist()}
        
        # Newton's method for equality-constrained minimization
        # min c^T x - sum(log(x_i)) s.t. Ax = b
        # KKT: c - 1/x + A^T nu = 0, Ax = b
        # Newton system (Schur complement):
        #   (A diag(x^2) A^T) dnu = -A (x^2 * g)
        #   dx = -x^2 * (g + A^T dnu)
        # where g = c - 1/x
        
        for iteration in range(100):
            inv_x = 1.0 / x
            g = c - inv_x  # gradient
            x2 = x * x
            
            # Schur complement system
            Ax2 = A * x2  # m x n
            S = Ax2 @ A.T  # m x m
            rhs = -Ax2 @ g  # m-vector
            
            try:
                L = cho_factor(S)
                dnu = cho_solve(L, rhs)
            except Exception:
                dnu = np.linalg.solve(S + 1e-12 * np.eye(m), rhs)
            
            dx = -x2 * (g + A.T @ dnu)
            
            # Newton decrement squared
            lam_sq = np.sum(dx * dx * inv_x * inv_x)
            if lam_sq / 2 < 1e-10:
                break
            
            # Maximum step size to maintain x > 0
            alpha = 1.0
            neg_dx = dx < 0
            if np.any(neg_dx):
                alpha = min(1.0, 0.99 * np.min(-x[neg_dx] / dx[neg_dx]))
            
            # Backtracking line search
            f0 = c @ x - np.sum(np.log(x))
            for _ in range(50):
                x_new = x + alpha * dx
                if np.all(x_new > 0):
                    f_new = c @ x_new - np.sum(np.log(x_new))
                    if f_new <= f0 - 0.01 * alpha * lam_sq:
                        break
                alpha *= 0.5
            
            x = x + alpha * dx
        
        return {"solution": x.tolist()}