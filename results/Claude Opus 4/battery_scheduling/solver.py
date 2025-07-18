import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Optimized battery scheduling solver using scipy linprog with sparse matrices.
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
        eta = float(battery["efficiency"])  # Battery efficiency
        eta_inv = 1.0 / eta
        
        # Variables: [q(T), c_in(T), c_out(T)]
        # Total: 3*T variables
        n_vars = 3 * T
        
        # Objective: minimize p^T * c_in - p^T * c_out
        c_obj = np.zeros(n_vars)
        c_obj[T:2*T] = p  # Coefficients for c_in
        c_obj[2*T:3*T] = -p  # Coefficients for c_out
        
        # Build sparse constraint matrices
        # Equality constraints: battery dynamics + cyclic constraint
        A_eq_rows = []
        b_eq = []
        
        # Battery dynamics: q[t+1] = q[t] + eta*c_in[t] - eta_inv*c_out[t]
        for t in range(T-1):
            row = np.zeros(n_vars)
            row[t+1] = 1.0  # q[t+1]
            row[t] = -1.0   # q[t]
            row[T+t] = -eta  # c_in[t]
            row[2*T+t] = eta_inv  # c_out[t]
            A_eq_rows.append(row)
            b_eq.append(0.0)
        
        # Cyclic constraint: q[0] = q[T-1] + eta*c_in[T-1] - eta_inv*c_out[T-1]
        row = np.zeros(n_vars)
        row[0] = 1.0  # q[0]
        row[T-1] = -1.0  # q[T-1]
        row[T+T-1] = -eta  # c_in[T-1]
        row[2*T+T-1] = eta_inv  # c_out[T-1]
        A_eq_rows.append(row)
        b_eq.append(0.0)
        
        A_eq = csr_matrix(np.array(A_eq_rows))
        b_eq = np.array(b_eq)
        
        # Inequality constraints: no power back to grid
        # u[t] + c_in[t] - c_out[t] >= 0
        # => -c_in[t] + c_out[t] <= u[t]
        A_ub_rows = []
        b_ub = []
        
        for t in range(T):
            row = np.zeros(n_vars)
            row[T+t] = -1.0  # c_in[t]
            row[2*T+t] = 1.0  # c_out[t]
            A_ub_rows.append(row)
            b_ub.append(u[t])
        
        A_ub = csr_matrix(np.array(A_ub_rows))
        b_ub = np.array(b_ub)
        
        # Variable bounds
        bounds = []
        # q: 0 <= q <= Q
        for i in range(T):
            bounds.append((0, Q))
        # c_in: 0 <= c_in <= C
        for i in range(T):
            bounds.append((0, C))
        # c_out: 0 <= c_out <= D
        for i in range(T):
            bounds.append((0, D))
        
        # Solve using HiGHS method (fastest for LP)
        result = linprog(
            c_obj, 
            A_ub=A_ub, 
            b_ub=b_ub,
            A_eq=A_eq, 
            b_eq=b_eq,
            bounds=bounds,
            method='highs',
            options={'presolve': True, 'disp': False}
        )
        
        if not result.success:
            return {"status": "not_optimal", "optimal": False}
        
        # Extract solution
        sol_vals = result.x
        
        q_val = sol_vals[:T]
        c_in_val = sol_vals[T:2*T]
        c_out_val = sol_vals[2*T:3*T]
        c_net = c_in_val - c_out_val
        
        # Calculate costs
        cost_without_battery = float(p @ u)
        cost_with_battery = float(p @ (u + c_net))
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