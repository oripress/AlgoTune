import numpy as np
from scipy.optimize import linprog
import highspy

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        try:
            # Parse input
            A = np.array(problem["a"], dtype=np.float64)
            b = np.array(problem["b"], dtype=np.float64)
            m, n = A.shape
            
            # Compute norms
            norms = np.linalg.norm(A, axis=1)
            
            # Highs setup
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            
            inf = 1e30
            
            # Variables: x (n), r (1)
            # Objective: minimize -r
            col_cost = np.zeros(n + 1, dtype=np.float64)
            col_cost[n] = -1.0
            
            col_lower = np.full(n + 1, -inf, dtype=np.float64)
            col_lower[n] = 0.0
            col_upper = np.full(n + 1, inf, dtype=np.float64)
            
            # Add variables (columns)
            # starts array size: num_cols + 1
            h.addCols(n + 1, col_cost, col_lower, col_upper, 0, np.zeros(n + 2, dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64))
            
            # Construct matrix [A, norms] for addRows
            # We construct the values array directly
            mat = np.empty((m, n + 1), dtype=np.float64)
            mat[:, :n] = A
            mat[:, n] = norms
            
            values = mat.ravel()
            
            # Indices: 0, 1, ..., n repeated m times
            col_indices = np.tile(np.arange(n + 1, dtype=np.int32), m)
            
            # Starts: 0, n+1, 2(n+1), ...
            starts = np.arange(0, (m + 1) * (n + 1), n + 1, dtype=np.int32)
            
            # Row bounds
            row_lower = np.full(m, -inf, dtype=np.float64)
            row_upper = b
            
            # Add rows
            h.addRows(m, row_lower, row_upper, len(values), starts, col_indices, values)
            
            h.run()
            
            sol = h.getSolution()
            return {"solution": list(sol.col_value)[:n]}
            
        except Exception:
            # Fallback
            A = np.array(problem["a"], dtype=np.float64)
            b = np.array(problem["b"], dtype=np.float64)
            m, n = A.shape
            norms = np.linalg.norm(A, axis=1)
            c = np.zeros(n + 1)
            c[-1] = -1.0
            A_ub = np.column_stack((A, norms))
            bounds = [(None, None)] * n + [(0, None)]
            res = linprog(c, A_ub=A_ub, b_ub=b, bounds=bounds, method='highs')
            return {"solution": res.x[:n].tolist() if res.x is not None else [0.0]*n}