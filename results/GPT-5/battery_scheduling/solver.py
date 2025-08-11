from typing import Any, Dict, List

import numpy as np
from scipy import sparse as sp
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the battery scheduling problem as a sparse linear program using HiGHS via SciPy.

        This minimizes p^T (u + c) subject to battery dynamics, capacity/rate limits,
        cyclic constraint, and no-export constraint.
        """
        # Extract problem parameters
        T = int(problem["T"])
        p = np.asarray(problem["p"], dtype=float)
        u = np.asarray(problem["u"], dtype=float)
        battery = problem["batteries"][0]  # Single battery

        # Battery parameters
        Q = float(battery["Q"])
        C = float(battery["C"])
        D = float(battery["D"])
        eta = float(battery["efficiency"])
        # Optional degradation cost
        deg_cost = float(problem.get("deg_cost", 0.0))

        # Variable ordering: x = [q(0:T), c_in(0:T), c_out(0:T)]  -> length 3T
        n = T
        q_start = 0
        ci_start = n
        co_start = 2 * n
        total_vars = 3 * n

        # Objective: minimize p^T (c_in - c_out)
        c_obj = np.zeros(total_vars, dtype=float)
        c_obj[ci_start:co_start] = p
        c_obj[co_start:] = -p

        # Equality constraints: battery dynamics with cyclic constraint
        # For t = 0..T-2: q[t+1] - q[t] - eta*c_in[t] + (1/eta)*c_out[t] = 0
        # For t = T-1:    q[0] - q[T-1] - eta*c_in[T-1] + (1/eta)*c_out[T-1] = 0
        row_idx: List[int] = []
        col_idx: List[int] = []
        data: List[float] = []
        if n >= 2:
            r = np.arange(n - 1)
            row_idx.extend(r.tolist() + r.tolist())
            col_idx.extend(r.tolist() + (r + 1).tolist())
            data.extend((-np.ones(n - 1)).tolist() + (np.ones(n - 1)).tolist())
        # Last row for cyclic
        row_idx.extend([n - 1, n - 1])
        col_idx.extend([n - 1, 0])
        data.extend([-1.0, 1.0])
        Aq = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n, n)).tocsc()

        Aci = (-eta) * sp.identity(n, format="csc")
        Aco = (1.0 / eta) * sp.identity(n, format="csc")
        A_eq = sp.hstack([Aq, Aci, Aco], format="csc")
        b_eq = np.zeros(n, dtype=float)

        # Inequality constraints: no export, c_in - c_out >= -u  ->  -c_in + c_out <= u
        Zq = sp.csc_matrix((n, n))
        Aub_ci = -sp.identity(n, format="csc")
        Aub_co = sp.identity(n, format="csc")
        A_ub = sp.hstack([Zq, Aub_ci, Aub_co], format="csc")
        b_ub = u.astype(float, copy=False)

        # Variable bounds
        bounds: List[tuple] = []
        bounds.extend([(0.0, Q)] * n)   # q
        bounds.extend([(0.0, C)] * n)   # c_in
        bounds.extend([(0.0, D)] * n)   # c_out

        # Solve LP using HiGHS
        res = linprog(
            c=c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not res.success or res.x is None:
            status_msg = getattr(res, "message", "solver_error")
            return {"status": status_msg, "optimal": False}

        x = res.x
        q = x[q_start:q_start + n]
        c_in = x[ci_start:co_start]
        c_out = x[co_start:co_start + n]
        c_net = c_in - c_out

        # Costs
        cost_without_battery = float(np.dot(p, u))
        cost_with_battery_energy = float(np.dot(p, u + c_net))
        degradation_cost = float(deg_cost * np.sum(c_in + c_out)) if deg_cost > 0 else 0.0
        cost_with_battery = cost_with_battery_energy + degradation_cost
        savings = cost_without_battery - cost_with_battery
        savings_percent = float(100.0 * savings / cost_without_battery) if cost_without_battery != 0 else 0.0

        result = {
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
            "savings_percent": savings_percent,
        }
        return result