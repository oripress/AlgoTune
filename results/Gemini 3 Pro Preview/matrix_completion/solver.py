import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        inds = np.array(problem["inds"])
        a = np.array(problem["a"])
        n = problem["n"]
        
        rows = inds[:, 0]
        cols = inds[:, 1]
        
        n_obs = len(a)
        n_missing = n * n - n_obs
        
        B = np.ones((n, n))
        B[rows, cols] = a
        
        if n_missing == 0:
            vals = np.linalg.eigvals(B)
            pf_val = np.max(np.abs(vals))
            return {"B": B.tolist(), "optimal_value": pf_val}

        u = np.ones(n) / np.sqrt(n)
        v = np.ones(n) / np.sqrt(n)
        
        lam = 0.0
        
        for it in range(200):
            # Power iteration for u
            for _ in range(10):
                u_next = B @ u
                norm_u = np.linalg.norm(u_next)
                if norm_u < 1e-20: break
                u = u_next / norm_u
            
            # Power iteration for v
            for _ in range(10):
                v_next = B.T @ v
                norm_v = np.linalg.norm(v_next)
                if norm_v < 1e-20: break
                v = v_next / norm_v
            
            # Estimate lambda
            Bu = B @ u
            lam = (v @ Bu) / (v @ u)
            
            u = np.abs(u)
            v = np.abs(v)
            u = np.maximum(u, 1e-20)
            v = np.maximum(v, 1e-20)
            
            log_v = np.log(v)
            log_u = np.log(u)
            
            sum_log_v = np.sum(log_v)
            sum_log_u = np.sum(log_u)
            
            sum_log_vu_all = n * sum_log_v + n * sum_log_u
            sum_log_vu_obs = np.sum(log_v[rows]) + np.sum(log_u[cols])
            
            sum_log_vu_missing = sum_log_vu_all - sum_log_vu_obs
            log_alpha = sum_log_vu_missing / n_missing
            
            # Update B
            # B_target = exp(log_alpha - log_v - log_u)
            # We can do this efficiently
            
            log_B_target = log_alpha - (log_v[:, None] + log_u[None, :])
            B_target = np.exp(log_B_target)
            
            # Reset observed
            B_target[rows, cols] = a
            
            diff = np.max(np.abs(B - B_target))
            B = B_target
            
            if diff < 1e-6:
                break
        
        return {"B": B.tolist(), "optimal_value": lam}