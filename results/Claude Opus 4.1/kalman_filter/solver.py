import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict) -> dict:
        """Solve Kalman filtering problem as a convex optimization."""
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        C = np.array(problem["C"])
        y = np.array(problem["y"])
        x0 = np.array(problem["x_initial"])
        tau = float(problem["tau"])
        
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]
        
        # Variables
        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")
        v = cp.Variable((N, m), name="v")
        
        # Objective
        obj = cp.Minimize(cp.sum_squares(w) + tau * cp.sum_squares(v))
        
        # Constraints
        cons = []
        cons.append(x[0] == x0)
        for t in range(N):
            cons.append(x[t + 1] == A @ x[t] + B @ w[t])
        for t in range(N):
            cons.append(y[t] == C @ x[t] + v[t])
        
        # Solve
        prob = cp.Problem(obj, cons)
        try:
            prob.solve()
        except:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        
        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }