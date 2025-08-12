import cvxpy as cp
import numpy as np

class Solver:
    def __init__(self):
        # Pre-configure solver settings for optimal performance
        self.solver_kwargs = {
            'verbose': False,
            'eps': 1e-6,
            'max_iters': 1000,
            'normalize': True,
            'scale': 10.0,
            'adaptive_scale': True,
            'alpha': 1.4,
        }
    
    def solve(self, problem: dict) -> dict:
        """
        Solve the optimal advertising problem using CVXPY with optimizations.
        
        :param problem: Dictionary with problem parameters
        :return: Dictionary with optimal displays and revenue
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
        D = cp.Variable((m, n))
        
        # Reformulate the objective using auxiliary variables
        z = cp.Variable(m)
        
        # Pre-compute P*R for efficiency
        PR = P * R.reshape(m, 1)
        
        # Define constraints using vectorized operations
        constraints = [
            D >= 0,  # Non-negative displays
            cp.sum(D, axis=0) <= T,  # Traffic capacity per time slot
            cp.sum(D, axis=1) >= c,  # Minimum display requirements
            z <= B,  # Budget constraints
        ]
        
        # Add revenue constraints using matrix multiplication
        # z[i] <= R[i] * sum(P[i, :] * D[i, :])
        constraints.append(z <= cp.sum(cp.multiply(PR, D), axis=1))
        
        # Define and solve the problem
        prob = cp.Problem(cp.Maximize(cp.sum(z)), constraints)
        
        try:
            # Try SCS solver first (often fastest for large problems)
            prob.solve(solver=cp.SCS, **self.solver_kwargs)
            
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Fallback to ECOS if SCS fails
                prob.solve(solver=cp.ECOS, verbose=False, max_iters=200)
                
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return {"status": prob.status, "optimal": False}
            
            # Calculate actual revenue using vectorized operations
            D_val = D.value
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