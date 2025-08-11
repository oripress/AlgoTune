import cvxpy as cp
import numpy as np
from scipy.special import xlogy
import math

class Solver:
    def solve(self, problem):
        # Direct access for speed
        P = np.asarray(problem["P"], dtype=np.float64)
        m, n = P.shape
        
        # Quick validation
        if not (n > 0 and m > 0):
            return None
            
        # Define optimization variables
        x = cp.Variable(shape=n)
        
        # Precompute c vector using numpy's vectorized operations
        # c_j = sum_{i=1..m} P_ij * log2(P_ij)
        log_P = np.log2(P, out=np.zeros_like(P), where=(P>0))
        c = np.sum(P * log_P, axis=0)
        
        # Define y = Px (output distribution)
        y = P @ x
        
        # Objective: maximize mutual information
        # Using the entropy formulation: c^T x + H(Y) where H(Y) = sum(y_i * log2(y_i))
        mutual_information = c @ x + cp.sum(cp.entr(y)) / np.log(2)
        
        # Define constraints
        constraints = [cp.sum(x) == 1, x >= 0]
        
        # Create and solve optimization problem with optimized settings
        prob = cp.Problem(cp.Maximize(mutual_information), constraints)
        
        # Try different solvers for better performance
        try:
            # Try OSQP first as it's often faster
            prob.solve(solver=cp.OSQP, verbose=False, max_iter=200,
                      eps_abs=1e-6, eps_rel=1e-6)
        except:
            # Fall back to ECOS with aggressive settings
            prob.solve(solver=cp.ECOS, verbose=False, max_iters=200,
                      abstol=1e-7, reltol=1e-7)
        
        # Return solution directly
        return {"x": x.value.tolist(), "C": float(prob.value)}