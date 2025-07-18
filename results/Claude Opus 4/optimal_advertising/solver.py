import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the optimal advertising problem using CVXPY with optimizations.
        
        Key optimizations:
        1. Reformulate the minimum operator using auxiliary variables
        2. Use ECOS solver with tight tolerances
        3. Vectorize operations
        4. Ensure constraints are properly satisfied
        """
        # Extract problem parameters
        P = np.array(problem["P"], dtype=np.float64)
        R = np.array(problem["R"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        c = np.array(problem["c"], dtype=np.float64)
        T = np.array(problem["T"], dtype=np.float64)
        
        # Derive m and n from P matrix
        m, n = P.shape
        
        # Define variables
        D = cp.Variable((m, n), nonneg=True)  # Non-negative constraint built-in
        z = cp.Variable(m, nonneg=True)  # Auxiliary variables for revenue
        
        # Define objective: maximize total revenue
        objective = cp.Maximize(cp.sum(z))
        
        # Define constraints
        constraints = []
        
        # Vectorized revenue constraints
        # z_i <= R_i * sum_t(P_it * D_it) for all i
        clicks_expr = cp.sum(cp.multiply(P, D), axis=1)
        constraints.append(z <= cp.multiply(R, clicks_expr))
        
        # Budget constraints: z_i <= B_i
        constraints.append(z <= B)
        
        # Traffic capacity per time slot (with small margin for numerical stability)
        constraints.append(cp.sum(D, axis=0) <= T)
        
        # Minimum display requirements
        constraints.append(cp.sum(D, axis=1) >= c)
        
        # Define and solve the problem
        prob = cp.Problem(objective, constraints)
        
        try:
            # Use ECOS with tight tolerances
            prob.solve(solver=cp.ECOS, verbose=False, 
                      abstol=1e-8, 
                      reltol=1e-8, 
                      feastol=1e-8,
                      max_iters=300)
            
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return {"status": prob.status, "optimal": False}
            
            # Get solution and ensure constraints are satisfied
            D_val = D.value
            
            # Post-process to ensure strict constraint satisfaction
            # Clip any small negative values to 0
            D_val = np.maximum(D_val, 0)
            
            # Ensure traffic capacity is not exceeded
            traffic_usage = np.sum(D_val, axis=0)
            for t in range(n):
                if traffic_usage[t] > T[t]:
                    # Scale down displays in this time slot
                    D_val[:, t] *= T[t] / traffic_usage[t]
            
            # Calculate actual revenue using vectorized operations
            clicks = np.sum(P * D_val, axis=1)
            revenue = np.minimum(R * clicks, B)
            
            # Return solution
            return {
                "status": prob.status,
                "optimal": True,
                "displays": D_val.tolist(),
                "clicks": clicks.tolist(),
                "revenue_per_ad": revenue.tolist(),
                "total_revenue": float(np.sum(revenue)),
                "objective_value": float(prob.value),
            }
            
        except cp.SolverError as e:
            return {"status": "solver_error", "optimal": False, "error": str(e)}
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}