import numpy as np
from typing import Any
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix, csc_matrix

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        fixed_costs = np.array(problem["fixed_costs"], dtype=np.float64)
        capacities = np.array(problem["capacities"], dtype=np.float64)
        demands = np.array(problem["demands"], dtype=np.float64)
        transportation_costs = np.array(problem["transportation_costs"], dtype=np.float64)
        
        n = len(fixed_costs)
        m = len(demands)
        
        # Variables: y_0..y_{n-1}, then x_{0,0}..x_{n-1,m-1}
        num_vars = n + n * m
        
        # Cost vector
        c = np.empty(num_vars, dtype=np.float64)
        c[:n] = fixed_costs
        c[n:] = transportation_costs.ravel()
        
        # Integrality: all binary (=1 means integer)
        integrality = np.ones(num_vars, dtype=int)
        
        # Bounds: all between 0 and 1
        bounds = Bounds(lb=0, ub=1)
        
        num_constraints = m + n + n * m
        
        # Pre-calculate total nnz
        nnz_type1 = n * m  # each of m rows has n entries  
        nnz_type2 = n * (1 + m)  # each of n rows has 1 + m entries
        nnz_type3 = n * m * 2  # each of n*m rows has 2 entries
        total_nnz = nnz_type1 + nnz_type2 + nnz_type3
        
        rows = np.empty(total_nnz, dtype=np.int32)
        cols = np.empty(total_nnz, dtype=np.int32)
        vals = np.empty(total_nnz, dtype=np.float64)
        
        idx = 0
        
        # Type 1: sum_i x_{ij} = 1 for each j  (m constraints, rows 0..m-1)
        # For each j, entries at columns n + i*m + j for i in range(n)
        jj, ii = np.meshgrid(np.arange(m), np.arange(n))
        r1 = jj.ravel()
        c1 = (n + ii * m + jj).ravel()
        v1 = np.ones(n * m, dtype=np.float64)
        end = idx + nnz_type1
        rows[idx:end] = r1
        cols[idx:end] = c1
        vals[idx:end] = v1
        idx = end
        
        # Type 2: sum_j d_j * x_{ij} - s_i * y_i <= 0 (n constraints, rows m..m+n-1)
        # For each i: one entry for y_i, then m entries for x_{ij}
        # y_i entries
        i_range = np.arange(n)
        r2_y = m + i_range
        c2_y = i_range
        v2_y = -capacities
        
        # x_{ij} entries
        ii2, jj2 = np.meshgrid(np.arange(n), np.arange(m), indexing='ij')
        r2_x = np.full(n * m, 0, dtype=np.int32)
        r2_x = (m + ii2).ravel()
        c2_x = (n + ii2 * m + jj2).ravel()
        v2_x = np.tile(demands, n)
        
        end = idx + n
        rows[idx:end] = r2_y
        cols[idx:end] = c2_y
        vals[idx:end] = v2_y
        idx = end
        
        end = idx + n * m
        rows[idx:end] = r2_x
        cols[idx:end] = c2_x
        vals[idx:end] = v2_x
        idx = end
        
        # Type 3: x_{ij} - y_i <= 0 (n*m constraints, rows m+n..m+n+n*m-1)
        # For each (i,j): two entries
        ii3, jj3 = np.meshgrid(np.arange(n), np.arange(m), indexing='ij')
        base_row = m + n + ii3 * m + jj3
        base_row_flat = base_row.ravel()
        
        # y_i entry (coefficient -1)
        r3_y = base_row_flat
        c3_y = ii3.ravel()
        v3_y = np.full(n * m, -1.0)
        
        # x_{ij} entry (coefficient 1)
        r3_x = base_row_flat
        c3_x = (n + ii3 * m + jj3).ravel()
        v3_x = np.ones(n * m, dtype=np.float64)
        
        end = idx + n * m
        rows[idx:end] = r3_y
        cols[idx:end] = c3_y
        vals[idx:end] = v3_y
        idx = end
        
        end = idx + n * m
        rows[idx:end] = r3_x
        cols[idx:end] = c3_x
        vals[idx:end] = v3_x
        idx = end
        
        A = csc_matrix((vals, (rows, cols)), shape=(num_constraints, num_vars))
        
        # Bounds for constraints
        lower = np.full(num_constraints, -np.inf)
        upper = np.full(num_constraints, np.inf)
        
        # Type 1: equality
        lower[:m] = 1.0
        upper[:m] = 1.0
        
        # Type 2: <= 0
        upper[m:m+n] = 0.0
        
        # Type 3: <= 0
        upper[m+n:] = 0.0
        
        constraints = LinearConstraint(A, lower, upper)
        
        result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if not result.success:
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n,
                "assignments": [[0.0] * m for _ in range(n)],
            }
        
        sol = result.x
        y_vals = np.clip(np.round(sol[:n]), 0, 1)
        x_vals = np.clip(np.round(sol[n:].reshape(n, m)), 0, 1)
        
        facility_status = [bool(v) for v in y_vals]
        assignments = x_vals.tolist()
        
        return {
            "objective_value": float(result.fun),
            "facility_status": facility_status,
            "assignments": assignments,
        }