import numpy as np
import highspy
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        fixed_costs = np.array(problem["fixed_costs"], dtype=np.float64)
        capacities = np.array(problem["capacities"], dtype=np.float64)
        demands = np.array(problem["demands"], dtype=np.float64)
        transportation_costs = np.array(problem["transportation_costs"], dtype=np.float64)
        
        n = fixed_costs.size
        m = demands.size
        
        # Problem dimensions
        num_col = n + n * m
        num_row = m + n + n * m
        
        # Objective
        col_cost = np.concatenate([fixed_costs, transportation_costs.flatten()])
        
        # Bounds
        col_lower = np.zeros(num_col, dtype=np.float64)
        col_upper = np.ones(num_col, dtype=np.float64)
        
        # Integrality
        integrality = [highspy.HighsVarType.kInteger] * num_col
        
        # Row bounds
        row_lower = np.concatenate([
            np.ones(m),                 # Demand
            np.full(n, -highspy.kHighsInf), # Capacity
            np.full(n*m, -highspy.kHighsInf) # Linking
        ])
        
        row_upper = np.concatenate([
            np.ones(m),                 # Demand
            np.zeros(n),                # Capacity
            np.zeros(n*m)               # Linking
        ])
        
        # Matrix construction (CSC)
        nnz = n * (1 + m) + n * m * 3
        
        start = np.zeros(num_col + 1, dtype=np.int32)
        index = np.zeros(nnz, dtype=np.int32)
        value = np.zeros(nnz, dtype=np.float64)
        
        current_idx = 0
        
        # y columns (0 to n-1)
        for i in range(n):
            start[i] = current_idx
            
            # Capacity
            index[current_idx] = m + i
            value[current_idx] = -capacities[i]
            current_idx += 1
            
            # Linking
            base_row = m + n + i * m
            indices = np.arange(base_row, base_row + m, dtype=np.int32)
            index[current_idx : current_idx + m] = indices
            value[current_idx : current_idx + m] = -1.0
            current_idx += m
            
        # x columns
        n_x = n * m
        x_starts = current_idx + np.arange(n_x + 1, dtype=np.int32) * 3
        start[n : n + n_x + 1] = x_starts
        
        demand_rows = np.tile(np.arange(m, dtype=np.int32), n)
        capacity_rows = np.repeat(np.arange(m, m + n, dtype=np.int32), m)
        linking_rows = np.arange(m + n, m + n + n * m, dtype=np.int32)
        
        rows_stacked = np.vstack((demand_rows, capacity_rows, linking_rows))
        x_indices_flat = rows_stacked.T.flatten()
        index[current_idx:] = x_indices_flat
        
        demands_repeated = np.tile(demands, n)
        vals_stacked = np.vstack((
            np.ones(n * m),
            demands_repeated,
            np.ones(n * m)
        ))
        x_values_flat = vals_stacked.T.flatten()
        value[current_idx:] = x_values_flat
        
        # Create Highs instance
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        
        # Ensure types
        col_cost = col_cost.astype(np.float64)
        col_lower = col_lower.astype(np.float64)
        col_upper = col_upper.astype(np.float64)
        row_lower = row_lower.astype(np.float64)
        row_upper = row_upper.astype(np.float64)
        start = start.astype(np.int32)
        index = index.astype(np.int32)
        value = value.astype(np.float64)
        
        # Corrected constructor call: only start, index, value
        # Use addRows and addCols instead of passModel/HighsSparseMatrix
        # Add empty rows
        h.addRows(row_lower, row_upper)
        
        # Add columns with matrix entries
        h.addCols(col_cost, col_lower, col_upper, start, index, value)
        
        # Set integrality
        integrality_array = [highspy.HighsVarType.kInteger] * num_col
        h.changeColsIntegrality(0, num_col - 1, integrality_array)
        
        if status != highspy.HighsStatus.kOk:
             return self._error_result(n, m)

        h.run()
        
        model_status = h.getModelStatus()
        if model_status not in [highspy.HighsModelStatus.kOptimal]:
             return self._error_result(n, m)
             
        sol = h.getSolution()
        info = h.getInfo()
        
        vals = np.array(sol.col_value)
        y_vals = vals[:n]
        x_vals = vals[n:].reshape((n, m))
        
        y_rounded = np.round(y_vals).astype(bool)
        x_rounded = np.round(x_vals)
        
        return {
            "objective_value": info.objective_function_value,
            "facility_status": y_rounded.tolist(),
            "assignments": x_rounded.tolist()
        }

    def _error_result(self, n, m):
        return {
            "objective_value": float("inf"),
            "facility_status": [False] * n,
            "assignments": [[0.0] * m for _ in range(n)],
        }