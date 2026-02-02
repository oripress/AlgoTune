import numpy as np
from typing import Any

try:
    from numba import njit
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

if USE_NUMBA:
    @njit
    def compute_variances(Q, M_inv):
        n, D = Q.shape
        variances = np.empty(n)
        for i in range(n):
            val = 0.0
            for r in range(D):
                tmp = 0.0
                for c in range(D):
                    tmp += M_inv[r, c] * Q[i, c]
                val += Q[i, r] * tmp
            variances[i] = val
        return variances

    @njit
    def _run_frank_wolfe(Q, M_inv, variances, u, max_iter, tol, D):
        n = Q.shape[0]
        for iter_count in range(max_iter):
            # Find max variance
            j = -1
            max_var = -1.0
            for i in range(n):
                if variances[i] > max_var:
                    max_var = variances[i]
                    j = i
            
            if max_var <= D + tol:
                break
                
            alpha = (max_var - D) / (D * (max_var - 1))
            
            # Update u
            for i in range(n):
                u[i] *= (1 - alpha)
            u[j] += alpha
            
            # z = M_inv @ Q[j]
            z = np.empty(D)
            for r in range(D):
                val = 0.0
                for c in range(D):
                    val += M_inv[r, c] * Q[j, c]
                z[r] = val
            
            gamma = alpha / (1 - alpha + alpha * max_var)
            scale = 1.0 / (1 - alpha)
            
            # Update variances
            for i in range(n):
                w_i = 0.0
                for k in range(D):
                    w_i += Q[i, k] * z[k]
                variances[i] = scale * (variances[i] - gamma * w_i * w_i)
            
            # Update M_inv
            for r in range(D):
                for c in range(D):
                    M_inv[r, c] = scale * (M_inv[r, c] - gamma * z[r] * z[c])
                    
        return u, M_inv

else:
    # Fallback implementations
    def compute_variances(Q, M_inv):
        return np.sum((Q @ M_inv) * Q, axis=1)

    def _run_frank_wolfe(Q, M_inv, variances, u, max_iter, tol, D):
        for _ in range(max_iter):
            j = np.argmax(variances)
            max_var = variances[j]
            if max_var <= D + tol:
                break
            alpha = (max_var - D) / (D * (max_var - 1))
            u *= (1 - alpha)
            u[j] += alpha
            z = M_inv @ Q[j]
            gamma = alpha / (1 - alpha + alpha * max_var)
            w = Q @ z
            scale = 1.0 / (1 - alpha)
            variances = scale * (variances - gamma * (w ** 2))
            M_inv = scale * (M_inv - gamma * np.outer(z, z))
        return u, M_inv

class Solver:
    def __init__(self):
        if USE_NUMBA:
            # Trigger compilation
            d_dummy = 2
            n_dummy = 5
            Q_dummy = np.ones((n_dummy, d_dummy+1))
            M_inv_dummy = np.eye(d_dummy+1)
            variances_dummy = np.ones(n_dummy)
            u_dummy = np.ones(n_dummy)/n_dummy
            compute_variances(Q_dummy, M_inv_dummy)
            _run_frank_wolfe(Q_dummy, M_inv_dummy, variances_dummy, u_dummy, 1, 1e-6, d_dummy+1)

    def solve(self, problem: dict[str, np.ndarray], **kwargs) -> dict[str, Any]:
        points = np.array(problem["points"])
        n, d = points.shape
        Q = np.column_stack((points, np.ones(n)))
        Q = np.ascontiguousarray(Q)
        D = d + 1
        
        # Smart initialization
        u = np.zeros(n)
        initial_indices = np.unique(np.concatenate((
            np.argmin(points, axis=0),
            np.argmax(points, axis=0)
        )))
        
        if len(initial_indices) < D:
            remaining = np.setdiff1d(np.arange(n), initial_indices)
            needed = D - len(initial_indices)
            if len(remaining) >= needed:
                initial_indices = np.concatenate((initial_indices, remaining[:needed]))
            else:
                initial_indices = np.arange(n)

        S = np.unique(initial_indices)
        u[S] = 1.0 / len(S)
        
        # Initial M_inv check
        try:
            M = (Q[S].T * u[S]) @ Q[S]
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            # Fallback to full uniform
            S = np.arange(n)
            u = np.ones(n) / n
            try:
                M = (Q.T * u) @ Q
                M_inv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                return {
                    "objective_value": float("inf"),
                    "ellipsoid": {"X": np.nan * np.ones((d, d)), "Y": np.nan * np.ones((d,))},
                }

        # Active Set Loop
        max_outer = 30
        for iter_outer in range(max_outer):
            # Prepare sub-problem
            Q_sub = Q[S]
            u_sub = u[S]
            # Renormalize u_sub to avoid drift
            u_sum = u_sub.sum()
            if u_sum > 1e-12:
                u_sub /= u_sum
            else:
                u_sub = np.ones(len(S)) / len(S)
            
            # Recompute M_inv from scratch for stability
            M_sub = (Q_sub.T * u_sub) @ Q_sub
            try:
                M_inv = np.linalg.inv(M_sub)
            except np.linalg.LinAlgError:
                break
            
            # Compute variances for sub-problem
            if USE_NUMBA:
                variances_sub = compute_variances(Q_sub, M_inv)
                # Run FW
                u_sub, M_inv = _run_frank_wolfe(Q_sub, M_inv, variances_sub, u_sub, 500, 1e-6, D)
            else:
                variances_sub = np.sum((Q_sub @ M_inv) * Q_sub, axis=1)
                u_sub, M_inv = _run_frank_wolfe(Q_sub, M_inv, variances_sub, u_sub, 500, 1e-6, D)
            
            # Update u
            u[S] = u_sub
            
            # Global check
            if USE_NUMBA:
                variances = compute_variances(Q, M_inv)
            else:
                variances = np.sum((Q @ M_inv) * Q, axis=1)
                
            max_var = np.max(variances)
            
            if max_var <= D + 1e-5:
                break
                
            # Add violators
            violators = np.argsort(variances)[-5:]
            new_violators = np.setdiff1d(violators, S)
            
            if len(new_violators) == 0:
                break
                
            S = np.concatenate((S, new_violators))

        # Reconstruct solution
        c = u @ points
        centered = points - c
        Sigma = (centered.T * u) @ centered
        
        try:
            vals, vecs = np.linalg.eigh(Sigma)
            if np.any(vals <= 1e-12):
                 return {
                    "objective_value": float("inf"),
                    "ellipsoid": {"X": np.nan * np.ones((d, d)), "Y": np.nan * np.ones((d,))},
                }
            
            inv_sqrt_vals = 1.0 / np.sqrt(vals)
            Sigma_inv_sqrt = vecs @ np.diag(inv_sqrt_vals) @ vecs.T
            
            X = (1.0 / np.sqrt(d)) * Sigma_inv_sqrt
            Y = -X @ c
            
            sign, logdet_Sigma = np.linalg.slogdet(Sigma)
            obj = (d / 2.0) * np.log(d) + 0.5 * logdet_Sigma
            
            # Ensure feasibility by scaling
            transformed_points = points @ X.T + Y
            norms = np.linalg.norm(transformed_points, axis=1)
            max_norm = np.max(norms)
            
            if max_norm > 1.0 + 1e-12:
                scale_factor = 1.0 / max_norm
                X *= scale_factor
                Y *= scale_factor
                obj -= d * np.log(scale_factor)

            return {
                "objective_value": obj,
                "ellipsoid": {
                    "X": X,
                    "Y": Y
                }
            }
            
        except np.linalg.LinAlgError:
             return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.nan * np.ones((d, d)), "Y": np.nan * np.ones((d,))},
            }