import numpy as np
import highspy

class Solver:
    def solve(self, problem, **kwargs):
        T = int(problem["T"])
        p = np.array(problem["p"], dtype=np.float64)
        u = np.array(problem["u"], dtype=np.float64)
        battery = problem["batteries"][0]
        
        Q = float(battery["Q"])
        C = float(battery["C"])
        D = float(battery["D"])
        eta = float(battery["efficiency"])
        inv_eta = 1.0 / eta
        
        n_vars = 3 * T
        n_constraints = 2 * T
        
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        
        # Build arrays
        lb = np.zeros(n_vars)
        ub = np.empty(n_vars)
        ub[:T] = C
        ub[T:2*T] = D
        ub[2*T:] = Q
        
        c_obj = np.empty(n_vars)
        c_obj[:T] = p
        c_obj[T:2*T] = -p
        c_obj[2*T:] = 0.0
        
        h.addVars(n_vars, lb, ub)
        h.changeColsCost(n_vars, np.arange(n_vars, dtype=np.int32), c_obj)
        
        # Row bounds
        row_lower = np.empty(n_constraints)
        row_lower[:T] = 0.0
        row_lower[T:] = -1e30
        row_upper = np.empty(n_constraints)
        row_upper[:T] = 0.0
        row_upper[T:] = u
        
        # Build sparse matrix in CSR format (row-wise) - fully vectorized
        # Dynamics: 4 entries per row (T rows)
        # Grid: 2 entries per row (T rows)
        nnz = 6 * T
        
        indices = np.empty(nnz, dtype=np.int32)
        values = np.empty(nnz)
        starts = np.empty(n_constraints + 1, dtype=np.int32)
        
        # Dynamics rows 0..T-2: indices [t, T+t, 2T+t, 2T+t+1], vals [-eta, inv_eta, -1, 1]
        t_arr = np.arange(T - 1, dtype=np.int32)
        base_dyn = 4 * t_arr
        indices[base_dyn] = t_arr
        indices[base_dyn + 1] = T + t_arr
        indices[base_dyn + 2] = 2 * T + t_arr
        indices[base_dyn + 3] = 2 * T + t_arr + 1
        values[base_dyn] = -eta
        values[base_dyn + 1] = inv_eta
        values[base_dyn + 2] = -1.0
        values[base_dyn + 3] = 1.0
        
        # Cyclic row T-1: indices [T-1, 2T-1, 2T, 3T-1], vals [-eta, inv_eta, 1, -1]
        cyc_base = 4 * (T - 1)
        indices[cyc_base:cyc_base+4] = [T - 1, 2 * T - 1, 2 * T, 3 * T - 1]
        values[cyc_base:cyc_base+4] = [-eta, inv_eta, 1.0, -1.0]
        
        # Grid rows T..2T-1: indices [t, T+t], vals [-1, 1]
        grid_t = np.arange(T, dtype=np.int32)
        grid_base = 4 * T + 2 * grid_t
        indices[grid_base] = grid_t
        indices[grid_base + 1] = T + grid_t
        values[grid_base] = -1.0
        values[grid_base + 1] = 1.0
        
        # Row starts
        starts[:T] = 4 * np.arange(T, dtype=np.int32)
        starts[T:2*T] = 4 * T + 2 * np.arange(T, dtype=np.int32)
        starts[n_constraints] = nnz
        
        h.addRows(n_constraints, row_lower, row_upper, nnz, starts, indices, values)
        h.run()
        
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return {"status": "not_optimal", "optimal": False}
        
        sol = h.getSolution()
        x = np.array(sol.col_value)
        
        c_in_val = x[:T]
        c_out_val = x[T:2*T]
        q_val = x[2*T:3*T]
        c_net = c_in_val - c_out_val
        
        cost_without_battery = float(p @ u)
        cost_with_battery = float(p @ (u + c_net))
        savings = cost_without_battery - cost_with_battery
        
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
            "savings_percent": float(100 * savings / cost_without_battery) if cost_without_battery > 0 else 0.0,
        }