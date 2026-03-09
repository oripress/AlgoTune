from __future__ import annotations

from typing import Any

import highspy
import numpy as np

class Solver:
    def __init__(self) -> None:
        self._template_cache: dict[
            tuple[int, float],
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}

    def _get_template(
        self, T: int, eta: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = (T, float(eta))
        tpl = self._template_cache.get(key)
        if tpl is not None:
            return tpl

        n = 3 * T
        m = 2 * T

        starts = np.empty(n + 1, dtype=np.int32)
        indices_list: list[int] = []
        values_list: list[float] = []
        pos = 0
        starts[0] = 0

        # q columns
        for j in range(T):
            if j == 0:
                indices_list.extend((T - 1, 0))
                values_list.extend((1.0, -1.0))
                pos += 2
            elif j == T - 1:
                indices_list.extend((T - 2, T - 1))
                values_list.extend((1.0, -1.0))
                pos += 2
            else:
                indices_list.extend((j - 1, j))
                values_list.extend((1.0, -1.0))
                pos += 2
            starts[j + 1] = pos

        # c_in columns
        for j in range(T):
            indices_list.extend((j, T + j))
            values_list.extend((-eta, -1.0))
            pos += 2
            starts[T + j + 1] = pos

        inv_eta = 1.0 / eta

        # c_out columns
        for j in range(T):
            indices_list.extend((j, T + j))
            values_list.extend((inv_eta, 1.0))
            pos += 2
            starts[2 * T + j + 1] = pos

        indices = np.asarray(indices_list, dtype=np.int32)
        values = np.asarray(values_list, dtype=float)
        row_lower = np.empty(m, dtype=float)
        row_upper = np.empty(m, dtype=float)
        row_lower[:T] = 0.0
        row_upper[:T] = 0.0
        row_lower[T:] = -highspy.kHighsInf
        row_upper[T:] = 0.0

        tpl = (starts, indices, values, row_lower, row_upper)
        self._template_cache[key] = tpl
        return tpl

    @staticmethod
    def _zero_solution(T: int, cost_without_battery: float) -> dict[str, Any]:
        z = np.zeros(T, dtype=float).tolist()
        return {
            "status": "optimal",
            "optimal": True,
            "battery_results": [
                {
                    "q": z,
                    "c": z,
                    "c_in": z,
                    "c_out": z,
                    "cost": cost_without_battery,
                }
            ],
            "total_charging": z,
            "cost_without_battery": cost_without_battery,
            "cost_with_battery": cost_without_battery,
            "savings": 0.0,
            "savings_percent": 0.0,
        }

    def solve(self, problem, **kwargs) -> Any:
        try:
            T = int(problem["T"])
            p = np.asarray(problem["p"], dtype=float)
            u = np.asarray(problem["u"], dtype=float)
            battery = problem["batteries"][0]

            Q = float(battery["Q"])
            C = float(battery["C"])
            D = float(battery["D"])
            eta = float(battery["efficiency"])

            cost_without_battery = float(p @ u)

            if T <= 0 or Q <= 0.0 or C <= 0.0 or D <= 0.0:
                return self._zero_solution(T, cost_without_battery)

            if np.all(p >= 0.0) and float(np.max(p)) * eta * eta <= float(np.min(p)) + 1e-12:
                return self._zero_solution(T, cost_without_battery)

            starts, indices, values, row_lower, row_upper = self._get_template(T, eta)

            highs = highspy.Highs()
            highs.setOptionValue("output_flag", False)
            highs.setOptionValue("threads", 1)
            highs.setOptionValue("presolve", "off")
            highs.setOptionValue("solver", "simplex")

            m = 2 * T
            n = 3 * T

            upper_rows = row_upper.copy()
            upper_rows[T:] = u

            empty_i32 = np.empty(0, dtype=np.int32)
            empty_f64 = np.empty(0, dtype=float)

            highs.addRows(m, row_lower, upper_rows, 0, empty_i32, empty_i32, empty_f64)

            col_cost = np.empty(n, dtype=float)
            col_cost[:T] = 0.0
            col_cost[T : 2 * T] = p
            col_cost[2 * T :] = -p

            col_lower = np.empty(n, dtype=float)
            col_upper = np.empty(n, dtype=float)
            col_lower[:T] = 0.0
            col_upper[:T] = Q
            col_lower[T : 2 * T] = 0.0
            col_upper[T : 2 * T] = C
            col_lower[2 * T :] = 0.0
            col_upper[2 * T :] = D

            highs.addCols(n, col_cost, col_lower, col_upper, len(values), starts, indices, values)
            highs.run()
            sol = highs.getSolution()
            x = np.asarray(sol.col_value, dtype=float)

            q = x[:T].copy()
            c_in = x[T : 2 * T].copy()
            c_out = x[2 * T :].copy()

            tol = 1e-10
            q[np.abs(q) < tol] = 0.0
            c_in[np.abs(c_in) < tol] = 0.0
            c_out[np.abs(c_out) < tol] = 0.0
            q[np.abs(q - Q) < tol] = Q
            c_in[np.abs(c_in - C) < tol] = C
            c_out[np.abs(c_out - D) < tol] = D

            c = c_in - c_out
            c[np.abs(c) < tol] = 0.0

            cost_with_battery = float(cost_without_battery + p @ c)
            savings = cost_without_battery - cost_with_battery

            return {
                "status": "optimal",
                "optimal": True,
                "battery_results": [
                    {
                        "q": q.tolist(),
                        "c": c.tolist(),
                        "c_in": c_in.tolist(),
                        "c_out": c_out.tolist(),
                        "cost": cost_with_battery,
                    }
                ],
                "total_charging": c.tolist(),
                "cost_without_battery": cost_without_battery,
                "cost_with_battery": cost_with_battery,
                "savings": savings,
                "savings_percent": (
                    float(100.0 * savings / cost_without_battery)
                    if cost_without_battery != 0.0
                    else 0.0
                ),
            }
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}