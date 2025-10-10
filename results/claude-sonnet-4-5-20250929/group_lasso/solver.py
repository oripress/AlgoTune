import cvxpy as cp
import numpy as np

class Solver:
    def solve(
        self, problem: dict[str, list[list[float]] | list[int] | float]
    ) -> dict[str, list[float] | float]:
        """Optimized logistic regression group lasso."""
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        gl = np.asarray(problem["gl"], dtype=np.int32)
        lba = problem["lba"]
        
        # Pre-compute group structure
        ulabels, inverseinds, pjs = np.unique(gl, return_inverse=True, return_counts=True)
        
        p = X.shape[1] - 1
        m = ulabels.shape[0]
        
        # Pre-compute group membership mask
        group_idx = np.zeros((p, m), dtype=bool)
        group_idx[np.arange(p), inverseinds] = True
        not_group_idx = ~group_idx
        
        sqr_group_sizes = np.sqrt(pjs)
        
        # Define variables
        beta = cp.Variable((p, m))
        beta0 = cp.Variable()
        
        # Reshape y for broadcasting
        y_col = y[:, None]
        
        # Extract features
        X_feat = X[:, 1:]
        
        # Define objective - match reference exactly
        linear_pred = cp.sum(X_feat @ beta, 1, keepdims=True) + beta0
        logreg = -cp.sum(cp.multiply(y_col, linear_pred)) + cp.sum(cp.logistic(cp.sum(X_feat @ beta, 1) + beta0))
        
        grouplasso = lba * cp.sum(cp.multiply(cp.norm(beta, 2, 0), sqr_group_sizes))
        objective = cp.Minimize(logreg + grouplasso)
        
        # Define constraints
        constraints = [beta[not_group_idx] == 0]
        
        # Solve problem
        prob = cp.Problem(objective, constraints)
        
        try:
            result = prob.solve()
        except:
            return None
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        
        if beta.value is None or beta0.value is None:
            return None
        
        # Extract beta values
        beta_result = beta.value[np.arange(p), inverseinds]
        
        return {
            "beta0": float(beta0.value),
            "beta": beta_result.tolist(),
            "optimal_value": float(result)
        }