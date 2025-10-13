from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs) -> Dict[str, Any]:
        """
        Solve the battery scheduling problem via a sparse LP using SciPy HiGHS.

        Decision variables (size 3T):
        - q[0:T]      : state of charge
        - c_in[0:T]   : charge rate   (>=0, <=C)
        - c_out[0:T]  : discharge rate(>=0, <=D)

        Constraints:
        - Dynamics:    q[t+1] - q[t] - eta*c_in[t] + (1/eta)*c_out[t] = 0 for t=0..T-2
        - Cyclic:      q[0] - q[T-1] - eta*c_in[T-1] + (1/eta)*c_out[T-1] = 0
        - Bounds:      0 <= q <= Q, 0 <= c_in <= C, 0 <= c_out <= D
        - No export:   u + (c_in - c_out) >= 0  =>  -c_in + c_out <= u

        Objective:
        - minimize p^T(c_in - c_out)  (equivalently total cost p^T(u + c))
        """
        # Extract problem parameters
        T = int(problem["T"])
        p = np.asarray(problem["p"], dtype=float)
        u = np.asarray(problem["u"], dtype=float)
        battery = problem["batteries"][0]  # single battery

        Q = float(battery["Q"])
        C = float(battery["C"])
        D = float(battery["D"])
        eta = float(battery["efficiency"])

        # Quick sanity checks
        assert p.shape[0] == T and u.shape[0] == T

        # LP variable layout: [q(0..T-1), c_in(0..T-1), c_out(0..T-1)]
        n_vars = 3 * T
        q_base = 0
        cin_base = T
        cout_base = 2 * T

        # Objective coefficients
        c_vec = np.zeros(n_vars, dtype=float)
        c_vec[cin_base : cin_base + T] = p
        c_vec[cout_base : cout_base + T] = -p

        # Bounds
        bounds: List[tuple[float, float]] = []
        # q bounds
        bounds.extend((0.0, Q) for _ in range(T))
        # c_in bounds
        bounds.extend((0.0, C) for _ in range(T))
        # c_out bounds
        bounds.extend((0.0, D) for _ in range(T))

        # Equality constraints (T rows): T-1 dynamics + 1 cyclic
        # Use sparse matrices
        try:
            from scipy.sparse import dok_matrix, csr_matrix  # type: ignore
            from scipy.optimize import linprog  # type: ignore
        except Exception as e:  # Fallback to CVXPY if SciPy unavailable
            return self._solve_cvxpy(problem)

        Aeq = dok_matrix((T, n_vars), dtype=float)
        beq = np.zeros(T, dtype=float)
        inv_eta = 1.0 / eta

        # Dynamics for t = 0..T-2
        for t in range(T - 1):
            Aeq[t, q_base + t + 1] = 1.0
            Aeq[t, q_base + t] = -1.0
            Aeq[t, cin_base + t] = -eta
            Aeq[t, cout_base + t] = inv_eta
            # beq[t] = 0

        # Cyclic at last row
        t = T - 1
        Aeq[t, q_base + 0] = 1.0
        Aeq[t, q_base + t] = -1.0
        Aeq[t, cin_base + t] = -eta
        Aeq[t, cout_base + t] = inv_eta
        # beq[t] = 0

        Aeq = csr_matrix(Aeq)

        # Inequality constraints: no export (-c_in + c_out <= u)
        Aub = dok_matrix((T, n_vars), dtype=float)
        bub = np.asarray(u, dtype=float)
        for t in range(T):
            Aub[t, cin_base + t] = -1.0
            Aub[t, cout_base + t] = 1.0
        Aub = csr_matrix(Aub)

        # Solve LP with HiGHS
        try:
            res = linprog(
                c=c_vec,
                A_ub=Aub,
                b_ub=bub,
                A_eq=Aeq,
                b_eq=beq,
                bounds=bounds,
                method="highs",
            )
        except Exception:
            # If the HiGHS call fails, fallback to CVXPY for robustness
            return self._solve_cvxpy(problem)

        if not res.success or res.x is None:
            # Fallback to CVXPY if HiGHS didn't succeed
            return self._solve_cvxpy(problem)

        x = res.x
        q = x[q_base : q_base + T]
        c_in = x[cin_base : cin_base + T]
        c_out = x[cout_base : cout_base + T]
        c_net = c_in - c_out

        # Costs
        cost_without = float(np.dot(p, u))
        cost_with = float(np.dot(p, u + c_net))
        savings = cost_without - cost_with
        savings_percent = 0.0 if cost_without == 0.0 else float(100.0 * savings / cost_without)

        # Prepare solution dictionary
        sol: Dict[str, Any] = {
            "status": "optimal" if res.success else res.message,
            "optimal": bool(res.success),
            "battery_results": [
                {
                    "q": q.tolist(),
                    "c": c_net.tolist(),
                    "c_in": c_in.tolist(),
                    "c_out": c_out.tolist(),
                    "cost": cost_with,
                }
            ],
            "total_charging": c_net.tolist(),
            "cost_without_battery": cost_without,
            "cost_with_battery": cost_with,
            "savings": savings,
            "savings_percent": savings_percent,
        }
        return sol

    def _solve_cvxpy(self, problem: dict) -> Dict[str, Any]:
        # Robust fallback using the reference CVXPY approach
        try:
            import cvxpy as cp  # type: ignore
            import numpy as np
        except Exception as e:
            return {"status": "solver_error", "optimal": False, "error": str(e)}

        T = int(problem["T"])
        p = np.array(problem["p"], dtype=float)
        u = np.array(problem["u"], dtype=float)
        battery = problem["batteries"][0]

        Q = float(battery["Q"])
        C = float(battery["C"])
        D = float(battery["D"])
        efficiency = float(battery["efficiency"])

        q = cp.Variable(T)
        c_in = cp.Variable(T)
        c_out = cp.Variable(T)
        c = c_in - c_out

        constraints = []
        constraints += [q >= 0, q <= Q]
        constraints += [c_in >= 0, c_in <= C]
        constraints += [c_out >= 0, c_out <= D]

        for t in range(T - 1):
            effective_charge = efficiency * c_in[t] - (1 / efficiency) * c_out[t]
            constraints.append(q[t + 1] == q[t] + effective_charge)

        effective_charge_last = efficiency * c_in[T - 1] - (1 / efficiency) * c_out[T - 1]
        constraints.append(q[0] == q[T - 1] + effective_charge_last)

        constraints.append(u + c >= 0)
        objective = cp.Minimize(p @ c)
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve()
        except Exception as e:
            return {"status": "solver_error", "optimal": False, "error": str(e)}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return {"status": prob.status, "optimal": False}

        c_net = c_in.value - c_out.value
        cost_without_battery = float(p @ u)
        cost_with_battery = float(p @ (u + c_net))
        savings = cost_without_battery - cost_with_battery

        return {
            "status": "optimal",
            "optimal": True,
            "battery_results": [
                {
                    "q": q.value.tolist(),
                    "c": c_net.tolist(),
                    "c_in": c_in.value.tolist(),
                    "c_out": c_out.value.tolist(),
                    "cost": cost_with_battery,
                }
            ],
            "total_charging": c_net.tolist(),
            "cost_without_battery": cost_without_battery,
            "cost_with_battery": cost_with_battery,
            "savings": savings,
            "savings_percent": float(100 * savings / cost_without_battery) if cost_without_battery != 0 else 0.0,
        }