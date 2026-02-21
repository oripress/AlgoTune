import numpy as np
import highspy
from scipy import sparse

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        T = int(problem["T"])
        p = np.array(problem["p"], dtype=np.float64)
        u = np.array(problem["u"], dtype=np.float64)
        battery = problem["batteries"][0]
        
        Q = float(battery["Q"])
        C = float(battery["C"])
        D = float(battery["D"])
        efficiency = float(battery["efficiency"])
        
        # Variables: q (T), c_in (T), c_out (T)
        num_vars = 3 * T
        num_cons = 2 * T
        
        col_cost = np.empty(num_vars, dtype=np.float64)
        col_cost[0:T] = 0.0
        col_cost[T:2*T] = p
        col_cost[2*T:3*T] = -p
        
        col_lower = np.zeros(num_vars, dtype=np.float64)
        
        col_upper = np.empty(num_vars, dtype=np.float64)
        col_upper[0:T] = Q
        col_upper[T:2*T] = C
        col_upper[2*T:3*T] = D
        
        row_lower = np.empty(num_cons, dtype=np.float64)
        row_lower[0:T] = 0.0
        row_lower[T:2*T] = -np.inf
        
        row_upper = np.empty(num_cons, dtype=np.float64)
        row_upper[0:T] = 0.0
        row_upper[T:2*T] = u
        
        # Build CSR matrix directly
        indptr = np.empty(num_cons + 1, dtype=np.int32)
        indptr[0:T+1] = np.arange(0, 4*T + 1, 4)
        indptr[T+1:2*T+1] = 4*T + np.arange(2, 2*T + 1, 2)
        
        nnz = 6 * T
        indices = np.empty(nnz, dtype=np.int32)
        data = np.empty(nnz, dtype=np.float64)
        
        # Equality constraints: q_{t+1} - q_t - eff * c_in_t + (1/eff) * c_out_t = 0
        idx = np.arange(T - 1)
        indices[0:4*(T-1):4] = idx
        indices[1:4*(T-1):4] = idx + 1
        indices[2:4*(T-1):4] = T + idx
        indices[3:4*(T-1):4] = 2*T + idx
        
        data[0:4*(T-1):4] = -1.0
        data[1:4*(T-1):4] = 1.0
        data[2:4*(T-1):4] = -efficiency
        data[3:4*(T-1):4] = 1.0 / efficiency
        
        # t = T - 1
        start = 4 * (T - 1)
        indices[start:start+4] = [0, T - 1, 2 * T - 1, 3 * T - 1]
        data[start:start+4] = [1.0, -1.0, -efficiency, 1.0 / efficiency]
        
        # Inequality constraints: -c_in_t + c_out_t <= u_t
        start = 4 * T
        idx2 = np.arange(T)
        indices[start:start+2*T:2] = T + idx2
        indices[start+1:start+2*T:2] = 2*T + idx2
        
        data[start:start+2*T:2] = -1.0
        data[start+1:start+2*T:2] = 1.0
        if not hasattr(self, 'h'):
            self.h = highspy.Highs()
            self.h.setOptionValue('output_flag', False)
            self.h.setOptionValue('presolve', 'off')
            self.h.setOptionValue('solver', 'simplex')
        else:
            self.h.clearModel()
        h = self.h
        
        h.addVars(num_vars, col_lower, col_upper)
        
        # Change objective sense to minimize (which is default, 1)
        h.changeColsCost(num_vars, np.arange(num_vars, dtype=np.int32), col_cost)
        
        # Add rows
        h.addRows(num_cons, row_lower, row_upper, nnz, indptr, indices, data)
        
        h.run()
        
        solution = h.getSolution()
        x = np.array(solution.col_value)
        
        q_val = x[:T]
        c_in_val = x[T:2*T]
        c_out_val = x[2*T:]
        c_net = c_in_val - c_out_val
        
        cost_without_battery = float(np.dot(p, u))
        cost_with_battery = float(np.dot(p, u + c_net))
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
            "savings_percent": float(100 * savings / cost_without_battery) if cost_without_battery != 0 else 0.0,
        }