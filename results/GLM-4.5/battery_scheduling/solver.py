import highspy
import numpy as np
from typing import Dict, Any

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the battery scheduling problem using HiGHS for maximum performance.

        This finds the optimal charging schedule for a battery that minimizes
        the total electricity cost over the time horizon.

        :param problem: Dictionary with problem parameters
        :return: Dictionary with optimal schedules and costs

        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        # Extract problem parameters
        T = int(problem["T"])
        p = np.array(problem["p"], dtype=np.float64)
        u = np.array(problem["u"], dtype=np.float64)
        battery = problem["batteries"][0]  # Single battery

        # Extract battery parameters
        Q = float(battery["Q"])  # Battery capacity
        C = float(battery["C"])  # Max charging rate
        D = float(battery["D"])  # Max discharging rate
        efficiency = float(battery["efficiency"])  # Battery efficiency
        inv_efficiency = 1.0 / efficiency  # Pre-calculate for efficiency

        # Create HiGHS instance
        h = highspy.Highs()
        
        # Set solver options for maximum performance
        h.setOptionValue('output_flag', False)
        h.setOptionValue('time_limit', 10.0)
        h.setOptionValue('presolve', 'on')
        h.setOptionValue('solver', 'simplex')
        h.setOptionValue('parallel', 'off')
        h.setOptionValue('threads', 1)
        h.setOptionValue('simplex_strategy', 'dual')
        h.setOptionValue('simplex_scale_strategy', 'aggressive')
        h.setOptionValue('ipm_optimality_tolerance', 1e-8)
        h.setOptionValue('ipm_feasibility_tolerance', 1e-8)
        
        # Pre-allocate arrays for batch operations
        # Number of variables: c_in[0..T-1], c_out[0..T-1], q[0..T-1]
        num_vars = 3 * T
        
        # Build constraint matrix in batch for efficiency
        # We'll use COO format (Coordinate format) for sparse matrix construction
        
        # 1. Add all variables at once
        # c_in: 0 <= c_in <= C
        for i in range(T):
            h.addCol(0.0, 0.0, C, 0, [], [])
        
        # c_out: 0 <= c_out <= D  
        for i in range(T):
            h.addCol(0.0, 0.0, D, 0, [], [])
            
        # q: 0 <= q <= Q
        for i in range(T):
            h.addCol(0.0, 0.0, Q, 0, [], [])
        
        # Set objective coefficients efficiently
        # Objective: minimize sum(p[t] * (c_in[t] - c_out[t]))
        for t in range(T):
            h.changeColCost(t, p[t])  # c_in variables
        for t in range(T):
            h.changeColCost(T + t, -p[t])  # c_out variables
        
        # Pre-allocate constraint arrays
        indices = np.empty(4, dtype=np.int32)
        values = np.empty(4, dtype=np.float64)
        
        # Add constraints efficiently
        # 1. No power back to grid: u[t] + c_in[t] - c_out[t] >= 0
        for t in range(T):
            lower_bound = -u[t]
            upper_bound = highspy.kHighsInf
            indices[0] = t
            indices[1] = T + t
            values[0] = 1.0
            values[1] = -1.0
            h.addRow(lower_bound, upper_bound, 2, indices, values)
        
        # 2. Battery dynamics: q[t+1] = q[t] + efficiency * c_in[t] - inv_efficiency * c_out[t]
        for t in range(T - 1):
            lower_bound = 0.0
            upper_bound = 0.0
            indices[0] = 2*T + t + 1  # q[t+1]
            indices[1] = 2*T + t      # q[t]
            indices[2] = t            # c_in[t]
            indices[3] = T + t        # c_out[t]
            values[0] = 1.0
            values[1] = -1.0
            values[2] = -efficiency
            values[3] = inv_efficiency
            h.addRow(lower_bound, upper_bound, 4, indices, values)
        
        # 3. Cyclic constraint: q[0] = q[T-1] + efficiency * c_in[T-1] - inv_efficiency * c_out[T-1]
        lower_bound = 0.0
        upper_bound = 0.0
        indices[0] = 2*T              # q[0]
        indices[1] = 2*T + T - 1     # q[T-1]
        indices[2] = T - 1            # c_in[T-1]
        indices[3] = T + T - 1        # c_out[T-1]
        values[0] = 1.0
        values[1] = -1.0
        values[2] = -efficiency
        values[3] = inv_efficiency
        h.addRow(lower_bound, upper_bound, 4, indices, values)
        
        # Solve the problem
        h.run()
        
        # Check if solution is optimal
        status = h.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            return {"status": str(status), "optimal": False}
        
        # Get solution values efficiently
        solution = h.getSolution()
        col_values = solution.col_value
        
        # Extract values using array slicing for efficiency
        c_in_val = np.array(col_values[:T])
        c_out_val = np.array(col_values[T:2*T])
        q_val = np.array(col_values[2*T:3*T])
        c_net = c_in_val - c_out_val

        # Calculate costs efficiently using np.dot
        cost_without_battery = float(np.dot(p, u))
        cost_with_battery = float(np.dot(p, u + c_net))
        savings = cost_without_battery - cost_with_battery

        # Return solution
        return {
            "status": "optimal",
            "optimal": True,
            "battery_results": [
                {
                    "q": q_val.tolist(),
                    "c": c_net.tolist(),
                    "c_in": c_in_val.tolist(),
                    "c_out": c_out_val.tolist(),
                    "cost": cost_with_battery,
                }
            ],
            "total_charging": c_net.tolist(),
            "cost_without_battery": cost_without_battery,
            "cost_with_battery": cost_with_battery,
            "savings": savings,
            "savings_percent": float(100 * savings / cost_without_battery),
        }