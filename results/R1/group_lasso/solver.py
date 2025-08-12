import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem, **kwargs):
        # Convert inputs to numpy arrays
        X = np.array(problem["X"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        gl = np.array(problem["gl"], dtype=np.int32)
        lba = float(problem["lba"])
        
        n, d = X.shape
        p = d - 1  # Features excluding intercept
        
        # Precompute groups EXACTLY as in reference implementation
        gl_reshaped = gl.reshape(-1, 1)  # Make it 2D: (p, 1)
        ulabels, inverseinds, pjs = np.unique(gl_reshaped, return_inverse=True, return_counts=True)
        m = len(ulabels)  # Number of unique groups
        
        # Create group index matrix EXACTLY as in reference
        group_idx = np.zeros((p, m), dtype=bool)
        group_idx[np.arange(p), inverseinds.flatten()] = True
        not_group_idx = ~group_idx
        sqr_group_sizes = np.sqrt(pjs).astype(np.float64)
        
        # Define variables EXACTLY as in reference
        beta = cp.Variable((p, m))
        beta0 = cp.Variable()
        
        # Reshape y to match reference (n,1)
        y = y.reshape(-1, 1)
        
        # Compute linear predictor EXACTLY as in reference
        linear_predictor = cp.sum(X[:, 1:] @ beta, axis=1, keepdims=True) + beta0
        
        # Define logistic loss EXACTLY as in reference
        log_loss = -cp.sum(cp.multiply(y, linear_predictor)) + cp.sum(cp.logistic(linear_predictor))
        
        # Define group lasso penalty EXACTLY as in reference
        group_penalty = lba * cp.sum(cp.multiply(cp.norm(beta, 2, axis=0), sqr_group_sizes))
        
        # Formulate problem EXACTLY as in reference
        objective = cp.Minimize(log_loss + group_penalty)
        constraints = [beta[not_group_idx] == 0]
        prob = cp.Problem(objective, constraints)
        
        # Solve EXACTLY as in reference
        try:
            result = prob.solve()
        except cp.SolverError as e:
            return None
        except Exception as e:
            return None
            
        # Check solver status EXACTLY as in reference
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
            
        if beta.value is None or beta0.value is None:
            return None
        
        # Extract solution EXACTLY as in reference
        beta_final = beta.value[np.arange(p), inverseinds.flatten()]
        
        return {
            "beta0": beta0.value,
            "beta": beta_final.tolist(),
            "optimal_value": result
        }