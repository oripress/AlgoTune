import numpy as np
import highspy

class Solver:
    def solve(self, problem, **kwargs):
        a = np.array(problem["a"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        m, n = a.shape
        
        # Precompute L2 norms for each constraint
        norms = np.linalg.norm(a, axis=1)
        
        # Create Highs instance
        h = highspy.Highs()
        
        # Disable output for better performance
        h.setOptionValue('output_flag', False)
        
        # Add variables: x (n variables) and r (1 variable)
        num_vars = n + 1
        lower_bounds = [-np.inf] * n + [0.0]
        upper_bounds = [np.inf] * (n + 1)
        h.addVars(num_vars, lower_bounds, upper_bounds)
        
        # Set objective: maximize r (which is the last variable)
        obj_coeffs = [0.0] * n + [1.0]
        h.changeColsCost(num_vars, list(range(num_vars)), obj_coeffs)
        
        # Add constraints
        for i in range(m):
            # Constraint: a_i^T x + r * ||a_i|| <= b_i
            row = []
            for j in range(n):
                row.append((j, float(a[i, j])))
            row.append((n, float(norms[i])))  # r coefficient
            h.addRow(-highspy.kHighsInf, float(b[i]), len(row), 
                     [idx for idx, _ in row], [val for _, val in row])
        
        # Set sense to maximization
        h.changeObjectiveSense(highspy.ObjSense.kMaximize)
        
        # Solve the problem
        h.run()
        
        # Get solution
        solution = h.getSolution()
        x_vals = solution.col_value[:n]
        
        return {'solution': x_vals}