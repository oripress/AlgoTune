from typing import Any, Dict
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        Solves the SVM task using cvxpy with the ECOS solver and hinge loss.

        After exhausting options by tuning solver parameters, this final attempt
        revisits the problem formulation. Instead of explicit slack variables,
        it uses cvxpy's `pos()` function to represent the hinge loss directly.
        This is a more compact formulation that can lead to different numerical
        behavior during the solver's canonocalization step. This is combined
        with the ECOS solver, which has proven to be the most robust option,
        using its default, well-balanced parameters. This represents the most
        plausible remaining path to matching the reference solution.
        """
        X = np.array(problem["X"])
        y = np.array(problem["y"])
        C = float(problem["C"])
        n_samples, n_features = X.shape

        # Define the optimization variables.
        beta = cp.Variable(n_features)
        beta0 = cp.Variable()

        # Define the objective function using the compact hinge loss formulation.
        loss = cp.sum(cp.pos(1 - cp.multiply(y, X @ beta + beta0)))
        regularization = 0.5 * cp.sum_squares(beta)
        objective = cp.Minimize(regularization + C * loss)

        # The problem has no other constraints with this formulation.
        prob = cp.Problem(objective)

        # Solve the problem using the ECOS solver with its default settings.
        prob.solve(solver=cp.ECOS, verbose=False)

        # If the solver fails or returns a non-optimal status, return None.
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return None

        beta_val = beta.value
        beta0_val = beta0.value

        # Handle cases where the solver reports success but returns None for values.
        if beta_val is None or beta0_val is None:
            return None

        optimal_value = prob.value

        # Calculate misclassification error.
        pred = X @ beta_val + beta0_val
        missclass_error = np.mean((pred * y) < 0)

        return {
            "beta0": float(beta0_val),
            "beta": beta_val.tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass_error),
        }