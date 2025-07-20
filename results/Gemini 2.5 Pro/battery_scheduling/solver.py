import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix
from typing import Any

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Solve the battery scheduling problem using scipy.optimize.linprog.
        This implementation uses vectorized sparse matrix construction and optimized
        data types for maximum performance.
        """
        # Extract problem parameters
        T = int(problem["T"])
        p = np.array(problem["p"], dtype=np.float64)
        u = np.array(problem["u"], dtype=np.float64)
        battery = problem["batteries"][0]

        # Extract battery parameters
        Q = float(battery["Q"])
        C = float(battery["C"])
        D = float(battery["D"])
        eta = float(battery["efficiency"])
        eta_c = eta
        inv_eta_d = 1.0 / eta

        # --- SciPy linprog formulation ---
        # Variables are concatenated: x = [q, c_in, c_out] (size 3T)
        
        # 1. Objective function: min(p^T * c_in - p^T * c_out)
        c_obj = np.concatenate([np.zeros(T, dtype=np.float64), p, -p])

        # 2. Bounds for variables: 0 <= q <= Q, 0 <= c_in <= C, 0 <= c_out <= D
        # Constructing a (n_vars, 2) array as required by scipy.linprog
        lb = np.zeros(3 * T, dtype=np.float64)
        ub = np.concatenate([
            np.full(T, Q, dtype=np.float64), 
            np.full(T, C, dtype=np.float64), 
            np.full(T, D, dtype=np.float64)
        ])
        bounds = np.column_stack((lb, ub))

        # 3. Inequality constraints: A_ub @ x <= b_ub
        # -c_in_t + c_out_t <= u_t  (equivalent to c_in_t - c_out_t >= -u_t)
        rng_T = np.arange(T)
        row_ub = np.concatenate([rng_T, rng_T])
        col_ub = np.concatenate([T + rng_T, 2 * T + rng_T])
        data_ub = np.concatenate([np.full(T, -1.0), np.full(T, 1.0)])
        A_ub = csc_matrix((data_ub, (row_ub, col_ub)), shape=(T, 3 * T))
        
        # 4. Equality constraints: A_eq @ x = b_eq
        rng_T_m1 = np.arange(T - 1)
        
        data_eq = np.concatenate([
            np.full(T - 1, -1.0), np.full(T - 1, 1.0), np.array([1.0, -1.0]),
            np.full(T, -eta_c), np.full(T, inv_eta_d)
        ])
        row_eq = np.concatenate([
            rng_T_m1, rng_T_m1, np.array([T - 1, T - 1]), rng_T, rng_T
        ])
        col_eq = np.concatenate([
            rng_T_m1, rng_T_m1 + 1, np.array([0, T - 1]), T + rng_T, 2 * T + rng_T
        ])

        A_eq = csc_matrix((data_eq, (row_eq, col_eq)), shape=(T, 3 * T))
        b_eq = np.zeros(T, dtype=np.float64)

        # Solve the LP using 'highs-ds' (dual simplex)
        res = linprog(c=c_obj, A_ub=A_ub, b_ub=u, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ds')

        # Process and return the solution
        if res.success:
            sol = res.x
            q_sol = sol[0:T]
            c_in_sol = sol[T:2*T]
            c_out_sol = sol[2*T:3*T]
            c_net = c_in_sol - c_out_sol

            cost_without_battery = float(np.dot(p, u))
            cost_with_battery = cost_without_battery + res.fun
            savings = -res.fun
            
            if abs(cost_without_battery) > 1e-9:
                savings_percent = 100.0 * savings / cost_without_battery
            else:
                savings_percent = 0.0

            return {
                "status": "optimal",
                "optimal": True,
                "battery_results": [{
                    "q": q_sol.tolist(),
                    "c": c_net.tolist(),
                    "c_in": c_in_sol.tolist(),
                    "c_out": c_out_sol.tolist(),
                    "cost": cost_with_battery,
                }],
                "total_charging": c_net.tolist(),
                "cost_without_battery": cost_without_battery,
                "cost_with_battery": cost_with_battery,
                "savings": savings,
                "savings_percent": savings_percent,
            }
        else:
            return {"status": res.message, "optimal": False}