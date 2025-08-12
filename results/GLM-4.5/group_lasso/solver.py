import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        """
        Solves the logistic regression group lasso using CVXPY.
        
        Args:
            problem: Dict containing X, y, gl, lba.
            
        Returns:
            Dict with estimates beta0, beta, optimal_value.
        """
        X = np.array(problem["X"])
        y = np.array(problem["y"])
        gl = np.array(problem["gl"])
        lba = problem["lba"]
        
        # Precompute group information more efficiently
        ulabels, inverseinds, pjs = np.unique(gl, return_inverse=True, return_counts=True)
        
        p = X.shape[1] - 1  # number of features
        m = ulabels.shape[0]  # number of unique groups
        
        # Create group mask more efficiently
        group_idx = np.zeros((p, m))
        group_idx[np.arange(p), inverseinds] = 1
        not_group_idx = np.logical_not(group_idx)
        
        
        sqr_group_sizes = np.sqrt(pjs)
        
        # --- Define CVXPY Variables ---
        beta = cp.Variable((p, m))
        beta0 = cp.Variable()
        lbacp = cp.Parameter(nonneg=True)
        y = y[:, None]
        
        # --- Define Objective ---
        #  g(β) + λ sum_{j=1}^J w_j || β_(j) ||_2^2
        #  g(β) = -sum_{i=1}^n [y_i (X β)_i] + sum_{i=1}^n log(1 + exp((X β)_i))
        # Compute Xβ more efficiently
        X_beta = X[:, 1:] @ beta
        X_beta_total = cp.sum(X_beta, axis=1) + beta0
        
        
        logreg = -cp.sum(cp.multiply(y.flatten(), X_beta_total)) + cp.sum(cp.logistic(X_beta_total))
        
        # Optimize group lasso computation
        group_norms = cp.norm(beta, 2, 0)
        grouplasso = lba * cp.sum(cp.multiply(group_norms, sqr_group_sizes))
        objective = cp.Minimize(logreg + grouplasso)
        
        # --- Define Constraints ---
        constraints = [beta[not_group_idx] == 0]
        lbacp.value = lba
        
        # --- Solve Problem ---
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve()
        except cp.SolverError as e:
            return None
        except Exception as e:
            return None
        
        # Check solver status
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        
        if beta.value is None or beta0.value is None:
            return None
        
        beta = beta.value[np.arange(p), inverseinds.flatten()]
        
        return {"beta0": beta0.value, "beta": beta.tolist(), "optimal_value": result}