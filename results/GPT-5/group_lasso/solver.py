from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np

class Solver:
    def solve(
        self, problem: dict[str, list[list[float]] | list[int] | float], **kwargs: Any
    ) -> dict[str, list[float] | float] | None:
        """
        Solves the logistic regression group lasso using CVXPY.

        Args:
            problem: Dict containing X, y, gl, lba.

        Returns:
            Dict with estimates beta0, beta, optimal_value.
        """
        try:
            X = np.array(problem["X"])
            y = np.array(problem["y"])
            gl = np.array(problem["gl"])
            lba = float(problem["lba"])
        except Exception:
            return None

        # Unique group labels and mapping from features to groups
        ulabels, inverseinds, pjs = np.unique(
            gl[:, None], return_inverse=True, return_counts=True
        )

        p = X.shape[1] - 1  # number of features (excluding intercept)
        m = ulabels.shape[0]  # number of unique groups

        # Build group indicator mask
        group_idx = np.zeros((p, m))
        group_idx[np.arange(p), inverseinds.flatten()] = 1
        not_group_idx = np.logical_not(group_idx)

        sqr_group_sizes = np.sqrt(pjs)

        # CVXPY variables
        beta = cp.Variable((p, m))
        beta0 = cp.Variable()
        y_col = y[:, None]

        # Aggregate beta across groups per feature to avoid forming n x m expressions
        v = cp.sum(beta, axis=1)  # shape (p,)

        # Logistic loss
        eta = X[:, 1:] @ v + beta0  # shape (n,)
        eta_col = cp.reshape(eta, (X.shape[0], 1))
        logreg = -cp.sum(cp.multiply(y_col, eta_col)) + cp.sum(cp.logistic(eta))

        # Group lasso penalty (unsquared group norms as in reference)
        grouplasso = lba * cp.sum(cp.multiply(cp.norm(beta, 2, 0), sqr_group_sizes))
        objective = cp.Minimize(logreg + grouplasso)

        # Constraints: zero out entries not in their group
        constraints = [beta[not_group_idx] == 0]

        # Solve
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve()
        except cp.SolverError:
            return None
        except Exception:
            return None

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None

        if beta.value is None or beta0.value is None:
            return None

        # Map matrix-form beta back to vector form
        beta_vec = beta.value[np.arange(p), inverseinds.flatten()]

        return {
            "beta0": float(beta0.value),
            "beta": beta_vec.tolist(),
            "optimal_value": float(result),
        }