import numpy as np
from scipy.optimize import minimize
from typing import Any

class Solver:
    def solve(self, problem: dict[str, np.ndarray], **kwargs) -> dict[str, Any]:
        points = np.array(problem["points"])
        n, d = points.shape
        
        P = np.hstack([points, np.ones((n, 1))]).T  # (d+1) x n
        
        def objective_and_grad(u):
            # Add small regularization to prevent singular matrix
            u_reg = np.maximum(u, 1e-10)
            Q = P @ (u_reg[:, np.newaxis] * P.T)
            
            try:
                # Cholesky decomposition is faster and more stable
                L = np.linalg.cholesky(Q)
                log_det = 2 * np.sum(np.log(np.diag(L)))
                
                # Compute Q^{-1} P
                # L L^T X = P => L Y = P => L^T X = Y
                Y = np.linalg.solve(L, P)
                Q_inv_P = np.linalg.solve(L.T, Y)
                
            except np.linalg.LinAlgError:
                # Fallback
                Q_inv = np.linalg.pinv(Q)
                sign, logdet = np.linalg.slogdet(Q)
                log_det = logdet if sign > 0 else -np.inf
                Q_inv_P = Q_inv @ P
                
            obj = -log_det
            
            # Gradient: -diag(P^T Q^{-1} P)
            grad = -np.sum(P * Q_inv_P, axis=0)
            
            return obj, grad

        # Initial guess: uniform distribution
        u0 = np.ones(n) / n
        
        # Bounds: u >= 0
        bounds = [(0, None) for _ in range(n)]
        
        # Constraints: sum(u) = 1
        constraints = {'type': 'eq', 'fun': lambda u: np.sum(u) - 1.0, 'jac': lambda u: np.ones(n)}
        
        res = minimize(
            objective_and_grad, 
            u0, 
            method='SLSQP', 
            jac=True, 
            bounds=bounds, 
            constraints=constraints,
            options={'ftol': 1e-8, 'maxiter': 1000, 'disp': False}
        )
        
        u_val = res.x
        u_val = np.maximum(u_val, 0)
        u_val /= np.sum(u_val)
        
        c = points.T @ u_val
        
        AUA = points.T @ (u_val[:, np.newaxis] * points)
        Sigma = AUA - np.outer(c, c)
        
        try:
            eigvals, eigvecs = np.linalg.eigh(Sigma)
            eigvals = np.maximum(eigvals, 1e-12)
            Sigma_inv_half = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            X = Sigma_inv_half / np.sqrt(d)
            Y = -X @ c
            
            max_norm = max(np.linalg.norm(X @ p + Y) for p in points)
            if max_norm > 1.0:
                X /= max_norm
                Y /= max_norm
            
            obj_val = -np.log(np.linalg.det(X))
            
            return {"objective_value": obj_val, "ellipsoid": {"X": X, "Y": Y}}
            
        except Exception as e:
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.nan * np.ones((d, d)), "Y": np.nan * np.ones((d,))},
            }