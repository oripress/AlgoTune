import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solves logistic regression with group lasso penalty:
            minimize g(β0,β) + λ ∑_j w_j ||β_j||_2
        where g is logistic loss and w_j = sqrt(p_j).
        """
        # Parse inputs
        X = np.array(problem["X"], dtype=float)   # (n, p+1)
        y = np.array(problem["y"], dtype=float)   # (n,)
        gl = np.array(problem["gl"], dtype=int)   # (p,)
        lba = float(problem["lba"])

        # Dimensions
        n, d = X.shape
        p = d - 1  # number of features

        # Unique groups and counts
        # gl is length p; make it column for unique
        ulabels, invinds, counts = np.unique(gl[:, None], return_inverse=True, return_counts=True)
        m = ulabels.shape[0]

        # Build mask to zero out cross-group entries
        group_idx = np.zeros((p, m), dtype=bool)
        group_idx[np.arange(p), invinds.flatten()] = True
        not_group_idx = ~group_idx

        # Group weights w_j = sqrt(p_j)
        w = np.sqrt(counts.astype(float))

        # CVXPY variables
        beta = cp.Variable((p, m))
        beta0 = cp.Variable()
        lbap = cp.Parameter(nonneg=True)

        # Logistic loss
        y_col = y[:, None]  # (n,1)
        linpred = cp.sum(X[:,1:] @ beta, axis=1, keepdims=True) + beta0  # (n,1)
        logreg = -cp.sum(cp.multiply(y_col, linpred)) + cp.sum(cp.logistic(cp.reshape(linpred, (n,))))

        # Group-lasso penalty
        grp_norms = cp.norm(beta, 2, axis=0)  # length m
        penalty = lbap * cp.sum(cp.multiply(w, grp_norms))

        # Constraints to enforce group structure
        constraints = [beta[not_group_idx] == 0]

        # Set regularization parameter
        lbap.value = lba

        # Solve problem
        prob = cp.Problem(cp.Minimize(logreg + penalty), constraints)
        result = prob.solve()

        # Check status
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        if beta.value is None or beta0.value is None:
            return None

        # Retrieve solution and convert to vector
        beta_mat = beta.value  # (p, m)
        beta_vec = beta_mat[np.arange(p), invinds.flatten()]

        return {
            "beta0": float(beta0.value),
            "beta": beta_vec.tolist(),
            "optimal_value": float(result),
        }