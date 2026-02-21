import cvxpy as cp
import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        X = np.array(problem["X"])
        y = np.array(problem["y"])
        gl = np.array(problem["gl"])
        lba = problem["lba"]

        ulabels, inverseinds, pjs = np.unique(gl[:, None], return_inverse=True, return_counts=True)

        p = X.shape[1] - 1  # number of features
        m = ulabels.shape[0]  # number of unique groups

        group_idx = np.zeros((p, m))
        group_idx[np.arange(p), inverseinds.flatten()] = 1
        not_group_idx = np.logical_not(group_idx)

        sqr_group_sizes = np.sqrt(pjs)

        # --- Define CVXPY Variables ---
        beta = cp.Variable((p, m))
        beta0 = cp.Variable()
        lbacp = cp.Parameter(nonneg=True)
        y = y[:, None]

        # --- Define Objective ---
        logreg = -cp.sum(
            cp.multiply(y, cp.sum(X[:, 1:] @ beta, 1, keepdims=True) + beta0)
        ) + cp.sum(cp.logistic(cp.sum(X[:, 1:] @ beta, 1) + beta0))

        grouplasso = lba * cp.sum(cp.multiply(cp.norm(beta, 2, 0), sqr_group_sizes))
        objective = cp.Minimize(logreg + grouplasso)

        # --- Define Constraints ---
        constraints = [beta[not_group_idx] == 0]
        lbacp.value = lba

        # --- Solve Problem ---
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        
        print("logreg:", logreg.value)
        print("grouplasso:", grouplasso.value)
        print("objective:", result)
        
        beta_val = beta.value[np.arange(p), inverseinds.flatten()]
        
        return {
            "beta0": float(beta0.value),
            "beta": beta_val.tolist(),
            "optimal_value": float(result)
        }