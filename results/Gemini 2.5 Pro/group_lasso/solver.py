import numpy as np
from scipy.special import expit

class Solver:
    def _logistic_loss(self, X, y, beta, beta0):
        X_beta = X[:, 1:] @ beta + beta0
        # log(1+exp(x)) can be computed stably with logaddexp(0, x)
        loss = -np.sum(y * X_beta) + np.sum(np.logaddexp(0, X_beta))
        return loss

    def _group_lasso_penalty(self, lba, w, group_indices, n_groups, beta):
        if beta.size == 0:
            return 0.0
        group_norms = np.sqrt(np.bincount(group_indices, weights=beta**2, minlength=n_groups))
        penalty = lba * np.sum(w * group_norms)
        return penalty

    def _objective(self, X, y, lba, w, group_indices, n_groups, beta, beta0):
        loss = self._logistic_loss(X, y, beta, beta0)
        penalty = self._group_lasso_penalty(lba, w, group_indices, n_groups, beta)
        return loss + penalty

    def solve(self, problem, **kwargs):
        X = np.array(problem["X"])
        y = np.array(problem["y"])
        gl = np.array(problem["gl"])
        lba = problem["lba"]

        n, p_plus_1 = X.shape
        p = p_plus_1 - 1

        if p == 0:
            # Only intercept. This is simple logistic regression on a constant.
            mean_y = np.mean(y)
            # Handle edge cases where mean_y is 0 or 1
            if mean_y <= 1e-9:
                beta0 = -20.0
            elif mean_y >= 1 - 1e-9:
                beta0 = 20.0
            else:
                beta0 = np.log(mean_y / (1 - mean_y))
            
            X_beta = np.full(n, beta0)
            opt_val = -np.sum(y * X_beta) + np.sum(np.logaddexp(0, X_beta))
            return {"beta0": beta0, "beta": [], "optimal_value": opt_val}

        # Group information
        group_labels, group_indices, group_sizes = np.unique(gl, return_inverse=True, return_counts=True)
        n_groups = len(group_labels)
        
        # Group weights
        w = np.sqrt(group_sizes)
        max_iter = kwargs.get("max_iter", 10000)
        tol = kwargs.get("tol", 1e-10)
        
        # FISTA variables
        x_k = np.zeros(p)
        x_old = np.zeros(p)
        b0_k = 0.0
        b0_old = 0.0
        t_k = 1.0
        t_old = 1.0

        # Lipschitz constant estimation (backtracking)
        L = 1.0
        eta = 1.5 # Use a less aggressive factor for stability

        # Main FISTA loop with restarts for stability
        for i in range(max_iter):
            # Extrapolation step
            t_ratio = (t_old - 1.0) / t_k
            y_beta = x_k + t_ratio * (x_k - x_old)
            y_b0 = b0_k + t_ratio * (b0_k - b0_old)

            # Gradient of smooth part (logistic loss) at the extrapolated point y
            X_beta_y = X[:, 1:] @ y_beta + y_b0
            p_i = expit(X_beta_y)
            grad_y_beta = X[:, 1:].T @ (p_i - y)
            grad_y_b0 = np.sum(p_i - y)
            f_y_smooth = self._logistic_loss(X, y, y_beta, y_b0)

            # Backtracking line search for L
            while True:
                # Proximal gradient step argument from y
                prox_arg_beta = y_beta - (1/L) * grad_y_beta
                
                # Proximal operator for group lasso
                thresholds = lba * w / L
                group_norms_prox = np.sqrt(np.bincount(group_indices, weights=prox_arg_beta**2, minlength=n_groups))
                inv_norms = np.zeros_like(group_norms_prox)
                nonzero_mask = group_norms_prox > 1e-12
                inv_norms[nonzero_mask] = 1.0 / group_norms_prox[nonzero_mask]
                multipliers = np.maximum(0, 1 - thresholds * inv_norms)
                scale_factors = multipliers[group_indices]
                x_next = prox_arg_beta * scale_factors
                
                b0_next = y_b0 - (1/L) * grad_y_b0

                # Check line search condition
                f_x_next_smooth = self._logistic_loss(X, y, x_next, b0_next)
                
                diff_beta_y = x_next - y_beta
                diff_b0_y = b0_next - y_b0
            t_next = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
            
            # Restart condition
            restart_check = (y_beta - x_next).T @ (x_next - x_k) + (y_b0 - b0_next) * (b0_next - b0_k)
            
            # Check for convergence using the state before the update
            diff_beta_norm = np.linalg.norm(x_next - x_k)
            diff_b0_norm = np.abs(b0_next - b0_k)
            
            # Update variables for next iteration
            x_old = x_k
            b0_old = b0_k
            x_k = x_next
            b0_k = b0_next

            # Correctly update momentum variables, with restart
            if restart_check > 0:
                t_old = 1.0
                t_k = 1.0
            else:
                t_old = t_k
                t_k = t_next

            if diff_beta_norm < tol * (1 + np.linalg.norm(x_k)) and diff_b0_norm < tol * (1 + np.abs(b0_k)):
                break
            t_old = t_k
            t_k = t_next

            if diff_beta_norm < tol * (1 + np.linalg.norm(x_k)) and diff_b0_norm < tol * (1 + np.abs(b0_k)):
                break
            diff_b0_norm = np.abs(b0_next - b0_k)
            
            # Update variables for next iteration
            x_old = x_k
            b0_old = b0_k
            x_k = x_next
            b0_k = b0_next
            t_old = t_k
            t_k = t_next

            if diff_beta_norm < tol * (1 + np.linalg.norm(x_k)) and diff_b0_norm < tol * (1 + np.abs(b0_k)):
                break
        beta = x_k
        beta0 = b0_k
        opt_val = self._objective(X, y, lba, w, group_indices, n_groups, beta, beta0)

        return {"beta0": beta0, "beta": beta.tolist(), "optimal_value": opt_val}