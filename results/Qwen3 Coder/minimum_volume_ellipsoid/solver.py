from typing import Any
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
import scipy.linalg

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Solves a given minimum volume covering ellipsoid problem using CVXPY.
        
        Args:
            problem: A dictionary with problem parameter:
                - points: list of given points to be contained in the ellipsoid.
        
        Returns:
            A dictionary containing the problem solution:
                - objective_value: the optimal objective value, which is proportional to logarithm of ellipsoid volume,
                - ellipsoid: a dictionary containing symmetric matrix X and ellipsoid center Y.
        """
        points = np.array(problem["points"])
        (n, d) = points.shape
        
        # For very small problems, try a direct approach
        # Removed direct computation for now to focus on CVXPY optimization
        
        # Define variables
        X = cp.Variable((d, d), symmetric=True)
        Y = cp.Variable((d,))
        # Define constraints using vectorized approach for better performance
        # Define constraints using all points
        constraints = [cp.SOC(1, X @ points[i] + Y) for i in range(n)]
        # Define objective: minimize -log(det(X))
        objective = cp.Minimize(-cp.log_det(X))
        
        # Create and solve the problem
        prob = cp.Problem(objective, constraints)
        
        try:
            # Try with CLARABEL first with optimized settings
            prob.solve(solver=cp.CLARABEL, verbose=False)
            
            # Check if a solution was found
            # Check if a solution was found
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                return {
                    "objective_value": float("inf"),
                    "ellipsoid": {"X": np.nan * np.ones((d, d)), "Y": np.nan * np.ones((d,))},
                }

            return {
                "objective_value": prob.value,
                "ellipsoid": {"X": X.value, "Y": Y.value}
            }
            
        except Exception as e:
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.nan * np.ones((d, d)), "Y": np.nan * np.ones((d,))},
            }
    def _direct_ellipsoid_computation(self, points):
        """Direct computation for very small problems"""
        try:
            # Use the circumscribed ellipsoid approach for tiny problems
            n, d = points.shape
            
            # Compute centroid
            centroid = np.mean(points, axis=0)
            
            # Center points around centroid
            centered_points = points - centroid
            
            # Compute covariance matrix
            cov_matrix = np.cov(centered_points.T)
            
            # Regularize to ensure positive definiteness
            cov_matrix += 1e-8 * np.eye(d)
            
            # Try to compute a reasonable ellipsoid
            # This is a heuristic approach for very small problems
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            # Scale to ensure all points are covered
            max_radius = np.max(np.linalg.norm(centered_points, axis=1))
            if max_radius > 0:
                scaling = 1.5 / max_radius  # Ensure coverage with margin
                X = scaling * np.eye(d)
                Y = -X @ centroid
                
                # Compute objective value
                obj_val = -np.log(np.linalg.det(X))
                
                # Check feasibility
                feasible = True
                for point in points:
                    if np.linalg.norm(X @ point + Y) > 1.01:  # Small tolerance
                        feasible = False
                        break
                
                if feasible:
                    return {
                        "objective_value": obj_val,
                        "ellipsoid": {"X": X, "Y": Y}
                    }
        except:
            pass
        return None