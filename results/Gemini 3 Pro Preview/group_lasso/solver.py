import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, list[list[float]] | list[int] | float]) -> dict[str, list[float] | float]:
        X = np.array(problem["X"])
        y = np.array(problem["y"])
        gl = np.array(problem["gl"])
        lba = problem["lba"]

        # Process group labels
        # gl is length p.
        # We want to map each feature to a group index 0..m-1
        ulabels, inverseinds, pjs = np.unique(gl, return_inverse=True, return_counts=True)
        
        p = X.shape[1] - 1  # number of features (excluding intercept column if present, though reference ignores col 0)
        m = ulabels.shape[0]  # number of unique groups

        # The reference implementation creates a (p, m) variable for beta
        # and constrains entries to be 0 if the feature doesn't belong to the group.
        
        group_idx = np.zeros((p, m), dtype=bool)
        group_idx[np.arange(p), inverseinds] = True
        not_group_idx = np.logical_not(group_idx)

        sqr_group_sizes = np.sqrt(pjs)

        # --- Define CVXPY Variables ---
        beta = cp.Variable((p, m))
        beta0 = cp.Variable()
        y_vec = y[:, None]

        # --- Define Objective ---
        # X[:, 1:] are the features.
        # (X[:, 1:] @ beta) results in (n, m). Summing over axis 1 gives (n,) vector of predictions?
        # Wait, if beta is (p, m) and we constrain it so only one entry per row is non-zero,
        # then (X[:, 1:] @ beta) is effectively X_features @ beta_vec if we collapsed beta.
        # But here it produces (n, m).
        # The reference sums over axis 1: cp.sum(X[:, 1:] @ beta, 1)
        
        linear_pred = cp.sum(X[:, 1:] @ beta, axis=1) + beta0
        
        logreg = -cp.sum(
            cp.multiply(y_vec, cp.reshape(linear_pred, (X.shape[0], 1)))
        ) + cp.sum(cp.logistic(linear_pred))

        # Group lasso penalty
        # norm(beta, 2, 0) computes L2 norm of each column (group).
        grouplasso = lba * cp.sum(cp.multiply(cp.norm(beta, 2, axis=0), sqr_group_sizes))
        
        objective = cp.Minimize(logreg + grouplasso)

        # --- Define Constraints ---
        constraints = [beta[not_group_idx] == 0]

        # --- Solve Problem ---
        prob = cp.Problem(objective, constraints)
        try:
            # Using ECOS or SCS usually. ECOS is good for SOCP.
            result = prob.solve()
        except cp.SolverError:
            return None
        except Exception:
            return None

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None

        if beta.value is None or beta0.value is None:
            return None

        # Extract beta vector
        beta_val = beta.value[np.arange(p), inverseinds]

        return {"beta0": beta0.value, "beta": beta_val.tolist(), "optimal_value": result}