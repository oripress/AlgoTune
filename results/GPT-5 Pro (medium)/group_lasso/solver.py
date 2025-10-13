from __future__ import annotations

from typing import Any, Dict, List

import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, List[List[float]] | List[int] | float], **kwargs) -> Dict[str, List[float] | float] | None:
        """
        Solves the logistic regression with group lasso penalty using CVXPY, matching the reference solution.

        Args:
            problem: Dict containing:
                - "X": list of lists (n x (p+1)) with first column as ones (intercept column)
                - "y": list of length n with labels in {0,1}
                - "gl": list of length p with group labels for each feature (1..J or any integers)
                - "lba": positive float, lambda

        Returns:
            Dict with keys:
             - "beta0": float (intercept)
             - "beta": list of floats of length p
             - "optimal_value": float objective value
            or None if the solver fails.
        """
        X = np.array(problem["X"])
        y = np.array(problem["y"])
        gl = np.array(problem["gl"])
        lba = float(problem["lba"])

        # Unique groups and mapping from feature index to group index
        ulabels, inverseinds, pjs = np.unique(gl[:, None], return_inverse=True, return_counts=True)

        p = X.shape[1] - 1  # number of features excluding intercept
        m = ulabels.shape[0]  # number of unique groups

        # Build mask to enforce a single non-zero per row (feature) in the corresponding group column
        group_idx = np.zeros((p, m), dtype=bool)
        group_idx[np.arange(p), inverseinds.flatten()] = True
        not_group_idx = np.logical_not(group_idx)

        sqr_group_sizes = np.sqrt(pjs)

        # Define CVXPY Variables
        beta = cp.Variable((p, m))
        beta0 = cp.Variable()
        y = y[:, None]  # shape (n,1) for broadcasting

        # Define Objective:
        # g(β) = -sum_i y_i * (Xβ)_i + sum_i log(1 + exp((Xβ)_i))
        # Penalize with λ * sum_j sqrt(p_j) * ||β_(j)||_2 (group lasso)
        X1 = X[:, 1:]  # features without intercept
        linear_term = cp.sum(X1 @ beta, 1, keepdims=True) + beta0
        logreg = -cp.sum(cp.multiply(y, linear_term)) + cp.sum(cp.logistic(cp.sum(X1 @ beta, 1) + beta0))
        grouplasso = lba * cp.sum(cp.multiply(cp.norm(beta, 2, 0), sqr_group_sizes))
        objective = cp.Minimize(logreg + grouplasso)

        # Constraints: zeros outside the assigned group column per feature
        constraints = [beta[not_group_idx] == 0]

        # Solve Problem
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve()
        except cp.SolverError:
            return None
        except Exception:
            return None

        if beta.value is None or beta0.value is None:
            return None

        # Extract vectorized beta corresponding to the true feature positions
        beta_vec = beta.value[np.arange(p), inverseinds.flatten()]

        return {
            "beta0": float(beta0.value),
            "beta": beta_vec.tolist(),
            "optimal_value": float(result),
        }