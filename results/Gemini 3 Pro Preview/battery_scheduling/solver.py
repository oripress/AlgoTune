import numpy as np
import highspy

class Solver:
    def solve(self, problem, **kwargs):
        T = int(problem["T"])
        p = np.array(problem["p"])
        u = np.array(problem["u"])
        battery = problem["batteries"][0]
        Q = float(battery["Q"])
        C = float(battery["C"])
        D = float(battery["D"])
        efficiency = float(battery["efficiency"])
        inv_efficiency = 1.0 / efficiency
        
        # Variables: q (0..T-1), c_in (T..2T-1), c_out (2T..3T-1)
        num_vars = 3 * T
        
        # Objective coefficients
        # Minimize p^T (c_in - c_out)
        col_cost = np.concatenate([np.zeros(T), p, -p])
        
        # Variable bounds
        col_lower = np.concatenate([np.zeros(T), np.zeros(T), np.zeros(T)])
        col_upper = np.concatenate([np.full(T, Q), np.full(T, C), np.full(T, D)])
        
        # Initialize Highs
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        
        # Add variables
        h.addVars(num_vars, col_lower, col_upper)
        h.changeColsCost(num_vars, np.arange(num_vars, dtype=np.int32), col_cost)
        
        # Constraints
        num_rows = 2 * T
        
        # Row bounds
        inf = highspy.kHighsInf
        row_lower = np.concatenate([np.zeros(T), np.full(T, -inf)])
        row_upper = np.concatenate([np.zeros(T), u])
        
        # Build constraint matrix
        indices = np.zeros(6*T, dtype=np.int32)
        values = np.zeros(6*T, dtype=np.float64)
        
        # Dynamics part (first 4*T entries)
        indices[0:4*T:4] = np.arange(T)
        values[0:4*T:4] = -1.0
        
        indices[1:4*T:4] = np.roll(np.arange(T), -1)
        values[1:4*T:4] = 1.0
        
        indices[2:4*T:4] = np.arange(T, 2*T)
        values[2:4*T:4] = -efficiency
        
        indices[3:4*T:4] = np.arange(2*T, 3*T)
        values[3:4*T:4] = inv_efficiency
        
        # Grid part (next 2*T entries)
        offset = 4*T
        indices[offset:6*T:2] = np.arange(T, 2*T)
        values[offset:6*T:2] = -1.0
        
        indices[offset+1:6*T:2] = np.arange(2*T, 3*T)
        values[offset+1:6*T:2] = 1.0
        
        # Start array
        start = np.zeros(num_rows + 1, dtype=np.int32)
        start[0:T+1] = np.arange(0, 4*T + 1, 4)
        start[T+1:] = np.arange(4*T + 2, 6*T + 1, 2)
        
        h.addRows(num_rows, row_lower, row_upper, len(values), start, indices, values)
        
        h.run()
        
        model_status = h.getModelStatus()
        if model_status != highspy.HighsModelStatus.kOptimal:
             return {"status": "solver_error", "optimal": False}
        
        # Extract solution
        sol = h.getSolution()
        x = np.array(sol.col_value)
        
        q = x[0:T]
        c_in = x[T:2*T]
        c_out = x[2*T:3*T]
        c_net = c_in - c_out
        
        cost_without_battery = float(np.dot(p, u))
        cost_with_battery = float(np.dot(p, u + c_net))
        savings = cost_without_battery - cost_with_battery
        
        return {
            "status": "optimal",
            "optimal": True,
            "battery_results": [
                {
                    "q": q.tolist(),
                    "c": c_net.tolist(),
                    "c_in": c_in.tolist(),
                    "c_out": c_out.tolist(),
                    "cost": cost_with_battery,
                }
            ],
            "total_charging": c_net.tolist(),
            "cost_without_battery": cost_without_battery,
            "cost_with_battery": cost_with_battery,
            "savings": savings,
            "savings_percent": float(100 * savings / cost_without_battery) if abs(cost_without_battery) > 1e-9 else 0.0,
        }