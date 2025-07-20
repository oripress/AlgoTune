from typing import Any, Dict, List
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List]:
        """
        Finds the center of the largest inscribed ball in a polyhedron P={x|a_i^T*x <= b_i}.
        This is the Chebyshev center problem, which can be formulated as a linear program.

        The LP formulation is:
        maximize r
        subject to: a_i^T * x_c + ||a_i||_2 * r <= b_i, for all i=1,...,m
                    r >= 0

        Variables are the center coordinates x_c (n) and the radius r (1).
        """
        # Convert input lists to numpy arrays for efficient computation
        a = np.array(problem["a"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        
        m, n = a.shape
        
        # The LP variables are [x_c_1, ..., x_c_n, r]. Total n+1 variables.
        # We want to maximize r, which is equivalent to minimizing -r.
        # The objective function vector `c` has a -1 for the 'r' variable.
        c = np.zeros(n + 1)
        c[n] = -1
        
        # Construct the inequality constraint matrix A_ub.
        # Each row corresponds to a constraint: a_i^T * x_c + ||a_i||_2 * r <= b_i
        A_ub = np.empty((m, n + 1), dtype=np.float64)
        A_ub[:, :n] = a
        
        # Calculate the L2 norm for each row of 'a' and place it in the last column of A_ub
        # A more efficient way to calculate row-wise L2 norms
        row_norms = np.sqrt(np.einsum('ij,ij->i', a, a))
        A_ub[:, n] = row_norms
        
        # Define bounds for the variables. x_c components are unbounded.
        # The radius 'r' must be non-negative.
        bounds = [(None, None)] * n + [(0, None)]
        
        # Solve the linear programming problem using SciPy's HiGHS dual simplex solver.
        # This method is generally efficient and robust for these types of problems.
        res = linprog(c, A_ub=A_ub, b_ub=b, bounds=bounds, method='highs-ds')
        
        # The optimal center x_c is the first n elements of the solution vector.
        x_c = res.x[:n]
        
        return {"solution": x_c.tolist()}