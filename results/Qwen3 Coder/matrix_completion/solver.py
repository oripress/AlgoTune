import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solves the Perron-Frobenius matrix completion using CVXPY.
        
        Args:
            problem: Dict containing inds, a, n.
            
        Returns:
            Dict with estimates B, optimal_value.
        """
        inds = problem["inds"]
        a = problem["a"]
        n = problem["n"]
        
        # Process indices and values more efficiently
        if len(inds) > 0:
            inds_array = np.asarray(inds, dtype=int)
            a_array = np.asarray(a, dtype=float)
            observed_rows, observed_cols = inds_array[:, 0], inds_array[:, 1]
        else:
            inds_array = np.empty((0, 2), dtype=int)
            a_array = np.empty(0, dtype=float)
            observed_rows, observed_cols = np.array([], dtype=int), np.array([], dtype=int)
        
        # Create matrix to track observed entries
        observed_mask = np.zeros((n, n), dtype=bool)
        if len(inds) > 0:
            observed_mask[observed_rows, observed_cols] = True
        
        # Find unobserved indices
        unobserved_mask = ~observed_mask
        
        # Create matrix to track observed entries
        observed_mask = np.zeros((n, n), dtype=bool)
        if len(inds_array) > 0:
            observed_mask[inds_array[:, 0], inds_array[:, 1]] = True
        
        # Find unobserved indices
        unobserved_mask = ~observed_mask
        otherinds = np.argwhere(unobserved_mask)
        
        # Define CVXPY Variables
        B = cp.Variable((n, n), pos=True)
        
        # Define Objective
        objective = cp.Minimize(cp.pf_eigenvalue(B))
        
        # Define Constraints
        constraints = []
        
        # Add product constraint for unobserved entries
        if np.any(unobserved_mask):
            constraints.append(cp.prod(B[unobserved_mask]) == 1.0)
        
        # Add equality constraints for observed entries
        if len(inds_array) > 0:
            constraints.append(B[inds_array[:, 0], inds_array[:, 1]] == a_array)
        
        # Solve Problem
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(gp=True, verbose=False)
        except (cp.SolverError, Exception):
            return None
        
        # Check solver status
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or B.value is None:
            return None
            
        return {
            "B": B.value.tolist(),
            "optimal_value": result,
        }