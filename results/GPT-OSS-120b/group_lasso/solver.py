import numpy as np
import cvxpy as cp
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve logistic regression with group‑lasso penalty using CVXPY.

        Parameters
        ----------
        problem : dict
            - "X": n × (p+1) matrix (first column is intercept of 1s)
            - "y": length‑n label vector (0/1)
            - "gl": length‑p list of group identifiers (one per non‑intercept feature)
            - "lba": regularisation parameter λ (float)

        Returns
        -------
        dict
            {"beta0": float, "beta": List[float], "optimal_value": float}
        """
        # ------------------------------------------------------------------
        # 1. Parse input
        # ------------------------------------------------------------------
        X = np.asarray(problem["X"], dtype=np.float64)          # n × (p+1)
        y = np.asarray(problem["y"], dtype=np.float64)        # n
        gl = np.asarray(problem["gl"], dtype=np.int64)        # p
        lam = float(problem["lba"])

        n, p_plus_one = X.shape
        p = p_plus_one - 1                                     # number of non‑intercept features
        X_no_intercept = X[:, 1:]                               # n × p

        # ------------------------------------------------------------------
        # 2. Group information
        # ------------------------------------------------------------------
        # Unique groups and mapping from feature index to group index
        unique_groups, inverse = np.unique(gl, return_inverse=True)
        m = unique_groups.shape[0]                                 # number of groups
        group_sizes = np.bincount(inverse)                         # p_j for each group
        w = np.sqrt(group_sizes)                                 # w_j = sqrt(p_j)

        # List of indices for each group (used for the regulariser)
        group_indices: List[np.ndarray] = [
            np.where(inverse == j)[0] for j in range(m)
        ]

        # ------------------------------------------------------------------
        # 3. CVXPY variables
        # ------------------------------------------------------------------
        beta = cp.Variable(p)          # coefficients for non‑intercept features
        beta0 = cp.Variable()        # intercept (no penalty)

        # ------------------------------------------------------------------
        # 4. Objective: logistic loss + λ * Σ w_j ||β_{group_j}||_2
        # ------------------------------------------------------------------
        # Logistic loss: -yᵀ (Xβ + β0) + Σ log(1 + exp(Xβ + β0))
        z = X_no_intercept @ beta + beta0
        logistic = -cp.sum(cp.multiply(y, z)) + cp.sum(cp.logistic(z))

        # Group‑lasso penalty
        penalty = 0
        for idx, wj in zip(group_indices, w):
            penalty += wj * cp.norm(beta[idx], 2)

        objective = cp.Minimize(logistic + lam * penalty)

        # ------------------------------------------------------------------
        # 5. Solve
        # ------------------------------------------------------------------
        prob = cp.Problem(objective)
        try:
            result = prob.solve()
        except Exception:
            # In case of solver failure, return None (will be treated as invalid)
            return None

        # ------------------------------------------------------------------
        # 6. Extract solution
        # ------------------------------------------------------------------
        beta_val = beta.value
        beta0_val = beta0.value

        # Ensure output dimensions match expectations
        if beta_val is None or beta0_val is None:
            return None

        # Post‑process: zero out negligible coefficients for exact match with reference
        tol = 1e-8
        beta_val = np.where(np.abs(beta_val) < tol, 0.0, beta_val)
        beta0_val = 0.0 if abs(beta0_val) < tol else beta0_val

        # Convert to Python types
        beta_list = beta_val.tolist()
        return {
            "beta0": float(beta0_val),
            "beta": beta_list,
            "optimal_value": float(result),
        }