import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem, **kwargs):
        n = 0
        try:
            # Extract and convert input data
            c = np.array(problem["c"])
            b = np.array(problem["b"])
            P = np.array(problem["P"])
            q = np.array(problem["q"])
            m = len(P)
            n = len(c)
            
            # Define optimization variables
            x = cp.Variable(n)
            
            # Build SOC constraints
            constraints = []
            for i in range(m):
                # SOC: ||P_i^T x||_2 <= b_i - q_i^T x
                constraints.append(cp.SOC(b[i] - q[i].T @ x, P[i].T @ x))
            
            # Formulate and solve problem
            prob = cp.Problem(cp.Minimize(c.T @ x), constraints)
            prob.solve(solver=cp.CLARABEL, verbose=False)
            
            # Handle solution status
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                return {"objective_value": float("inf"), "x": np.full(n, np.nan)}
                
            # Return solution with x as a numpy array
            return {
                "objective_value": float(prob.value),
                "x": np.array(x.value)
            }
                
        except Exception:
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}