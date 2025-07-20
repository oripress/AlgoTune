import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        # Parse input parameters
        T = int(problem.get("T", 0))
        p = np.array(problem.get("p", []), dtype=float)
        u = np.array(problem.get("u", []), dtype=float)
        if T == 0 or p.size != T or u.size != T:
            return {"status": "error", "optimal": False, "error": "Invalid dimensions"}
        b = problem["batteries"][0]
        Q = float(b["Q"])
        C = float(b["C"])
        D = float(b["D"])
        eff = float(b["efficiency"])

        # Number of variables: c_in (T), c_out (T), q (T)
        n = 3 * T

        # Objective: minimize sum_t p[t] * (c_in[t] - c_out[t])
        # Vector form c = [p, -p, 0]
        c_obj = np.concatenate((p, -p, np.zeros(T, dtype=float)))

        # Bounds for variables
        bounds = [(0.0, C)] * T + [(0.0, D)] * T + [(0.0, Q)] * T

        # Equality constraints: battery dynamics + cyclic
        # A_eq x = 0
        row_eq = []
        col_eq = []
        data_eq = []
        b_eq = np.zeros(T, dtype=float)

        for t in range(T):
            if t < T - 1:
                # q[t+1] - q[t] - eff*c_in[t] + (1/eff)*c_out[t] = 0
                row_eq += [t, t, t, t]
                col_eq += [t, T + t, 2*T + t, 2*T + t + 1]
                data_eq += [-eff, 1.0/eff, -1.0, 1.0]
            else:
                # cyclic: q[0] - q[T-1] - eff*c_in[T-1] + (1/eff)*c_out[T-1] = 0
                row_eq += [t, t, t, t]
                col_eq += [t, T + t, 2*T + t, 2*T]
                data_eq += [-eff, 1.0/eff, -1.0, 1.0]

        A_eq = coo_matrix((data_eq, (row_eq, col_eq)), shape=(T, n))

        # Inequality constraints: -c_in[t] + c_out[t] <= u[t]
        row_ub = []
        col_ub = []
        data_ub = []
        b_ub = u.copy()

        for t in range(T):
            row_ub += [t, t]
            col_ub += [t, T + t]
            data_ub += [-1.0, 1.0]

        A_ub = coo_matrix((data_ub, (row_ub, col_ub)), shape=(T, n))

        # Solve LP with HiGHS
        res = linprog(
            c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs',
        )

        if not res.success:
            return {"status": "infeasible", "optimal": False}

        x = res.x
        c_in = x[:T]
        c_out = x[T:2*T]
        q = x[2*T:3*T]
        c_net = (c_in - c_out).tolist()

        # Calculate costs
        cost_without = float(np.dot(p, u))
        cost_with = float(np.dot(p, u + (c_in - c_out)))
        savings = cost_without - cost_with
        savings_percent = 100.0 * savings / cost_without if cost_without != 0 else 0.0

        return {
            "battery_results": [
                {"q": q.tolist(), "c": c_net, "c_in": c_in.tolist(), "c_out": c_out.tolist(), "cost": cost_with}
            ],
            "total_charging": c_net,
            "cost_without_battery": cost_without,
            "cost_with_battery": cost_with,
            "savings": savings,
            "savings_percent": savings_percent,
            "optimal": True,
            "status": "optimal",
        }