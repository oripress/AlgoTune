import logging
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: Dict) -> Dict:
        """
        Solve the battery scheduling problem using a fast linear programming formulation
        with SciPy's high-performance HiGHS solver.
        Returns a dictionary matching the required output format.
        """
        try:
            # ------------------------------------------------------------------
            # Extract problem data
            # ------------------------------------------------------------------
            T = int(problem["T"])
            p = np.array(problem["p"], dtype=float)
            u = np.array(problem["u"], dtype=float)

            battery = problem["batteries"][0]  # single battery
            Q = float(battery["Q"])
            C = float(battery["C"])
            D = float(battery["D"])
            eta = float(battery["efficiency"])

            # ------------------------------------------------------------------
            # Variable ordering: [c_in (0..T-1), c_out (T..2T-1), q (2T..3T-1)]
            # ------------------------------------------------------------------
            n = 3 * T

            # Objective: minimize p·(u + c_in - c_out)
            # Constant term p·u can be added later; we only need coefficients for variables.
            c_obj = np.concatenate([p, -p, np.zeros(T)])

            # ------------------------------------------------------------------
            # Equality constraints (battery dynamics + cyclic)
            # ------------------------------------------------------------------
            # Each row corresponds to: q[t+1] - q[t] - eta*c_in[t] + (1/eta)*c_out[t] = 0
            A_eq = np.zeros((T, n))
            b_eq = np.zeros(T)

            for t in range(T - 1):
                # q[t+1] coefficient
                A_eq[t, 2 * T + (t + 1)] = 1.0
                # q[t] coefficient
                A_eq[t, 2 * T + t] = -1.0
                # c_in[t] coefficient
                A_eq[t, t] = -eta
                # c_out[t] coefficient
                A_eq[t, T + t] = 1.0 / eta

            # Cyclic condition (t = T-1)
            A_eq[T - 1, 2 * T + 0] = 1.0          # q[0]
            A_eq[T - 1, 2 * T + (T - 1)] = -1.0   # -q[T-1]
            A_eq[T - 1, T - 1] = -eta             # -eta * c_in[T-1]
            A_eq[T - 1, 2 * T - 1] = 1.0 / eta    # (1/eta) * c_out[T-1]

            # ------------------------------------------------------------------
            # Inequality constraints (no power back to grid)
            #   -c_in[t] + c_out[t] <= u[t]
            # ------------------------------------------------------------------
            A_ub = np.zeros((T, n))
            b_ub = np.copy(u)

            for t in range(T):
                A_ub[t, t] = -1.0          # -c_in[t]
                A_ub[t, T + t] = 1.0       # +c_out[t]

            # ------------------------------------------------------------------
            # Variable bounds
            # ------------------------------------------------------------------
            bounds = []
            # c_in bounds
            for _ in range(T):
                bounds.append((0.0, C))
            # c_out bounds
            for _ in range(T):
                bounds.append((0.0, D))
            # q bounds
            for _ in range(T):
                bounds.append((0.0, Q))

            # ------------------------------------------------------------------
            # Solve LP using HiGHS (default method for linprog)
            # ------------------------------------------------------------------
            res = linprog(
                c=c_obj,
                A_eq=A_eq,
                b_eq=b_eq,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method="highs",
                options={"presolve": True, "time_limit": 30},
            )

            if not res.success:
                return {"status": res.message, "optimal": False}

            # ------------------------------------------------------------------
            # Extract solution components
            # ------------------------------------------------------------------
            sol = res.x
            c_in_val = sol[0:T]
            c_out_val = sol[T : 2 * T]
            net_val = c_in_val - c_out_val
            q_val = sol[2 * T : 3 * T]

            # ------------------------------------------------------------------
            # Cost calculations
            # ------------------------------------------------------------------
            cost_without = float(p @ u)
            cost_with = float(p @ (u + net_val))
            savings = cost_without - cost_with

            # ------------------------------------------------------------------
            # Assemble output dictionary
            # ------------------------------------------------------------------
            result = {
                "status": "optimal",
                "optimal": True,
                "battery_results": [
                    {
                        "q": q_val.tolist(),
                        "c": net_val.tolist(),
                        "c_in": c_in_val.tolist(),
                        "c_out": c_out_val.tolist(),
                        "cost": cost_with,
                    }
                ],
                "total_charging": net_val.tolist(),
                "cost_without_battery": cost_without,
                "cost_with_battery": cost_with,
                "savings": savings,
                "savings_percent": float(100.0 * savings / cost_without),
            }
            return result

        except Exception as e:
            logging.exception("Error in Solver.solve")
            return {"status": "error", "optimal": False, "error": str(e)}