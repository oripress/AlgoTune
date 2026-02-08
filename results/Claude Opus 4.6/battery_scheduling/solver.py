import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

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
        
        # Variables: [c_in(T), c_out(T), q(T)] = 3T variables
        n_vars = 3 * T
        
        # Objective
        c_obj = np.zeros(n_vars)
        c_obj[:T] = p
        c_obj[T:2*T] = -p
        
        t_range = np.arange(T, dtype=np.int32)
        next_t = np.empty(T, dtype=np.int32)
        next_t[:T-1] = t_range[1:]
        next_t[T-1] = 0
        
        n_rows = 2 * T
        
        # Build COO arrays for entire constraint matrix
        # Eq rows: 4T entries, Ub rows: 2T entries = 6T total
        rows = np.empty(6 * T, dtype=np.int32)
        cols = np.empty(6 * T, dtype=np.int32)
        vals = np.empty(6 * T, dtype=np.float64)
        
        # Equality: rows 0..T-1
        i = 0
        rows[i:i+T] = t_range; cols[i:i+T] = t_range; vals[i:i+T] = -eta; i += T
        rows[i:i+T] = t_range; cols[i:i+T] = T + t_range; vals[i:i+T] = inv_eta; i += T
        rows[i:i+T] = t_range; cols[i:i+T] = 2*T + t_range; vals[i:i+T] = -1.0; i += T
        rows[i:i+T] = t_range; cols[i:i+T] = 2*T + next_t; vals[i:i+T] = 1.0; i += T
        
        # Inequality: rows T..2T-1
        rows[i:i+T] = T + t_range; cols[i:i+T] = t_range; vals[i:i+T] = -1.0; i += T
        rows[i:i+T] = T + t_range; cols[i:i+T] = T + t_range; vals[i:i+T] = 1.0; i += T
        
        A = csc_matrix((vals, (rows, cols)), shape=(n_rows, n_vars))
        A_eq = A[:T, :]
        A_ub = A[T:, :]
        b_eq = np.zeros(T)
        b_ub = u
        
        # Bounds as array for speed
        lb = np.zeros(n_vars)
        ub = np.empty(n_vars)
        ub[:T] = C
        ub[T:2*T] = D
        ub[2*T:] = Q
        
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=np.column_stack((lb, ub)), method='highs-ds',
                        options={'presolve': True, 'dual_feasibility_tolerance': 1e-7,
                                 'primal_feasibility_tolerance': 1e-7})
        
        if result.status != 0:
            return {"status": "solver_error", "optimal": False}
        
        x = result.x
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
            "savings_percent": float(100 * savings / cost_without_battery) if cost_without_battery != 0 else 0.0,
        }