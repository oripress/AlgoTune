import numpy as np
from scipy.linalg import eig

class Solver:
    def _power_iteration(self, B, n, tol=1e-9, max_iter=100):
        """
        Computes the right and left Perron-Frobenius eigenvectors using power iteration.
        Returns (right_eigenvector, left_eigenvector).
        """
        right_vec = np.random.rand(n)
        right_vec /= np.linalg.norm(right_vec)
        left_vec = np.random.rand(n)
        left_vec /= np.linalg.norm(left_vec)

        for _ in range(max_iter):
            right_vec_old, left_vec_old = right_vec, left_vec
            right_vec_new = B @ right_vec
            right_norm = np.linalg.norm(right_vec_new)
            if right_norm < 1e-30: return None, None
            left_vec_new = B.T @ left_vec
            left_norm = np.linalg.norm(left_vec_new)
            if left_norm < 1e-30: return None, None
            
            right_vec = right_vec_new / right_norm
            left_vec = left_vec_new / left_norm
            
            if np.allclose(right_vec, right_vec_old, atol=tol) and \
               np.allclose(left_vec, left_vec_old, atol=tol):
                break
        
        return right_vec, left_vec

    def solve(self, problem, **kwargs) -> dict:
        inds = np.array(problem["inds"])
        a = np.array(problem["a"])
        n = problem["n"]

        if len(inds) > 0 and np.any(a <= 0):
            return None

        known_mask = np.zeros((n, n), dtype=bool)
        if len(inds) > 0:
            known_mask[inds[:, 0], inds[:, 1]] = True
        
        unknown_mask = ~known_mask
        num_unknown = np.sum(unknown_mask)

        B = np.ones((n, n))
        if len(inds) > 0:
            # Initialize with the geometric mean of known entries for better scale.
            geo_mean = np.exp(np.mean(np.log(a)))
            B.fill(geo_mean)
            B[inds[:, 0], inds[:, 1]] = a
        
        if num_unknown == 0:
            try:
                evals, _ = eig(B)
                return {"B": B.tolist(), "optimal_value": float(np.max(np.abs(evals)))}
            except np.linalg.LinAlgError:
                return None

        max_iter = 500
        tol = 1e-9
        damping_factor = 0.5

        for i in range(max_iter):
            B_old = B.copy()

            v, u = self._power_iteration(B, n)
            if u is None:
                try:
                    evals, vl, vr = eig(B, left=True, right=True)
                    pf_idx = np.argmax(np.abs(evals))
                    v = np.real(vr[:, pf_idx])
                    u = np.real(vl[:, pf_idx])
                except np.linalg.LinAlgError:
                    return None
            
            u = np.maximum(np.abs(u), 1e-50)
            v = np.maximum(np.abs(v), 1e-50)

            log_u = np.log(u)
            log_v = np.log(v)
            
            log_prod_outer = np.add.outer(log_u, log_v)
            
            # Get the previous state in log-space for damping and scale preservation.
            log_B_old_unknown = np.log(np.maximum(B_old[unknown_mask], 1e-50))
            
            # Define the target B by preserving the geometric mean of the old B's unknown entries.
            # This is the critical correction to the update rule.
            mean_log_B_old = np.mean(log_B_old_unknown)
            mean_log_prod = np.mean(log_prod_outer[unknown_mask])
            log_B_target_unknown = mean_log_B_old + mean_log_prod - log_prod_outer[unknown_mask]
            
            # Apply damping between the old B and the new target B.
            log_B_new_unknown = (1 - damping_factor) * log_B_old_unknown + \
                                damping_factor * log_B_target_unknown
            
            B[unknown_mask] = np.exp(log_B_new_unknown)

            norm_diff = np.linalg.norm(B - B_old)
            norm_old = np.linalg.norm(B_old)
            if norm_old > 1e-9 and (norm_diff / norm_old) < tol:
                break
            elif norm_diff < tol:
                break
        
        try:
            evals, _ = eig(B)
            optimal_value = np.max(np.abs(evals))
        except np.linalg.LinAlgError:
            return None

        return {
            "B": B.tolist(),
            "optimal_value": float(optimal_value),
        }