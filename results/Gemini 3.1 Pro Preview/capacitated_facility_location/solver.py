import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import milp, Bounds, LinearConstraint

class Solver:
    def solve(self, problem: dict, **kwargs):
        fixed_costs = np.array(problem["fixed_costs"], dtype=float)
        capacities = np.array(problem["capacities"], dtype=float)
        demands = np.array(problem["demands"], dtype=float)
        transportation_costs = np.array(problem["transportation_costs"], dtype=float)
        
        n = len(fixed_costs)
        m = len(demands)
        
        num_vars = n + n * m
        num_constraints = m + n + n * m
        
        # Objective
        c = np.empty(num_vars, dtype=float)
        c[:n] = fixed_costs
        c[n:] = transportation_costs.ravel()
        
        # Pre-allocate arrays for COO matrix
        nnz = n * m + n + n * m + n * m + n * m
        row = np.empty(nnz, dtype=int)
        col = np.empty(nnz, dtype=int)
        data = np.empty(nnz, dtype=float)
        
        idx = 0
        
        # 1. sum_{i} x_{ij} = 1
        size1 = n * m
        row[idx:idx+size1] = np.repeat(np.arange(m), n)
        col[idx:idx+size1] = n + np.arange(n * m).reshape(n, m).T.ravel()
        data[idx:idx+size1] = 1.0
        idx += size1
        
        # 2. sum_{j} d_j * x_{ij} - s_i * y_i <= 0
        size2_y = n
        row[idx:idx+size2_y] = m + np.arange(n)
        col[idx:idx+size2_y] = np.arange(n)
        data[idx:idx+size2_y] = -capacities
        idx += size2_y
        
        size2_x = n * m
        row[idx:idx+size2_x] = m + np.repeat(np.arange(n), m)
        col[idx:idx+size2_x] = n + np.arange(n * m)
        data[idx:idx+size2_x] = np.tile(demands, n)
        idx += size2_x
        
        # 3. x_{ij} - y_i <= 0
        size3_y = n * m
        row[idx:idx+size3_y] = m + n + np.arange(n * m)
        col[idx:idx+size3_y] = np.repeat(np.arange(n), m)
        data[idx:idx+size3_y] = -1.0
        idx += size3_y
        
        size3_x = n * m
        row[idx:idx+size3_x] = m + n + np.arange(n * m)
        col[idx:idx+size3_x] = n + np.arange(n * m)
        data[idx:idx+size3_x] = 1.0
        idx += size3_x
        
        A = coo_matrix((data, (row, col)), shape=(num_constraints, num_vars))
        
        lb = np.empty(num_constraints, dtype=float)
        lb[:m] = 1.0
        lb[m:] = -np.inf
        
        ub = np.empty(num_constraints, dtype=float)
        ub[:m] = 1.0
        ub[m:] = 0.0
        
        constraints = LinearConstraint(A, lb, ub)
        
        integrality = np.ones(num_vars)
        bounds = Bounds(0, 1)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success:
            y = np.round(res.x[:n]).astype(int)
            x = np.round(res.x[n:]).reshape((n, m)).astype(int)
            return {
                "objective_value": float(res.fun),
                "facility_status": y.astype(bool).tolist(),
                "assignments": x.tolist()
            }
        else:
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n,
                "assignments": [[0.0] * m for _ in range(n)],
            }