import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solves the logistic regression group lasso using CVXPY reference implementation.
        """
        # Load data
        X = np.array(problem["X"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        gl = np.array(problem["gl"], dtype=np.int64)
        lba = float(problem["lba"])

        # Unique group labels and counts
        ulabels, inverseinds, pjs = np.unique(gl[:, None], return_inverse=True, return_counts=True)

        # Dimensions
        n, D = X.shape
        p = D - 1  # number of features
        m = ulabels.shape[0]  # number of unique groups

        # Build group index mask for beta
        group_idx = np.zeros((p, m), dtype=bool)
        group_idx[np.arange(p), inverseinds.flatten()] = True
        not_group_idx = ~group_idx

        # Weights: sqrt of group sizes
        sqr_group_sizes = np.sqrt(pjs)

        # CVXPY Variables
        beta = cp.Variable((p, m))
        beta0 = cp.Variable()
        lbacp = cp.Parameter(nonneg=True)

        # Reshape y to column vector
        y_col = y[:, None]

        # Compute linear predictions: contributions sum over groups
        sumXB = X[:, 1:] @ beta  # shape (n, m)
        sumXB_col = cp.sum(sumXB, axis=1, keepdims=True)  # shape (n, 1)

        # Logistic loss g(beta)
        logreg = -cp.sum(cp.multiply(y_col, sumXB_col + beta0)) \
                 + cp.sum(cp.logistic(cp.sum(sumXB, axis=1) + beta0))

        # Group lasso penalty
        grouplasso = lba * cp.sum(cp.multiply(cp.norm(beta, 2, axis=0), sqr_group_sizes))

        # Objective and constraints
        objective = cp.Minimize(logreg + grouplasso)
        constraints = [beta[not_group_idx] == 0]
        lbacp.value = lba

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve()
        except cp.SolverError:
            return None
        except Exception:
            return None

        # Validate solution
        if beta.value is None or beta0.value is None:
            return None

        # Convert beta matrix to vector form
        beta_vec = beta.value[np.arange(p), inverseinds.flatten()]

        return {
            "beta0": float(beta0.value),
            "beta": beta_vec.tolist(),
            "optimal_value": float(result),
        }