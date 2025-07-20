import numpy as np
import cvxpy as cp

class Solver:
    def solve(
        self,
        problem: dict,
    ) -> dict:
        """
        Optimized SVM solver using primal formulation with OSQP solver.
        """
        X = np.array(problem["X"])
        y = np.array(problem["y"])
        C = float(problem["C"])
        n, p = X.shape
        
        # Primal variables
        beta = cp.Variable(p)
        beta0 = cp.Variable()
        xi = cp.Variable(n)
        
        # Objective: 1/2 ||β||^2 + C sum(ξ)
        objective = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))
        
        # Constraints: ξ ≥ 0, y_i(x_i^T β + β0) ≥ 1 - ξ_i
        constraints = [
            xi >= 0,
            cp.multiply(y, X @ beta + beta0) >= 1 - xi
        ]
        
        # Solve using OSQP which is optimized for QP problems
        prob = cp.Problem(objective, constraints)
        try:
            # OSQP is generally faster than ECOS for QP problems
            optimal_value = prob.solve(solver=cp.OSQP, verbose=False)
        except Exception as e:
            return None
            
        if beta.value is None or beta0.value is None:
            return None
            
        # Compute misclassification error
        pred = X @ beta.value + beta0.value
        misclass_error = np.mean((pred * y) < 0)
        
        return {
            "beta0": float(beta0.value),
            "beta": beta.value.tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(misclass_error),
        }