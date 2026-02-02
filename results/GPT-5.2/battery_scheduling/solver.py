from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

class Solver:
    """
    Fast solver for the single-battery scheduling LP using HiGHS (highspy).

    The reference uses CVXPY, which has high modeling overhead. This solver builds
    a sparse banded LP directly and solves it with HiGHS.
    """

    def __init__(self) -> None:
        self._highs = None
        self._highspy = None
        self._inf = 1e30
        # Cache A-matrix structure for (T, efficiency) since only those affect A.
        self._amat_cache: Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        try:
            import highspy  # type: ignore

            self._highspy = highspy
            self._inf = float(getattr(highspy, "kHighsInf", self._inf))
            h = highspy.Highs()
            # Silence output for speed.
            try:
                h.setOptionValue("output_flag", False)
            except Exception:
                pass
            # Prefer simplex for accurate feasibility on equalities.
            try:
                h.setOptionValue("solver", "simplex")
            except Exception:
                pass
            # Tighten feasibility tolerance to pass strict validator.
            for opt, val in (
                ("primal_feasibility_tolerance", 1e-9),
                ("dual_feasibility_tolerance", 1e-9),
            ):
                try:
                    h.setOptionValue(opt, val)
                except Exception:
                    pass

            self._highs = h
        except Exception:
            # Fallback path (should be rare): will use CVXPY in solve().
            self._highs = None
            self._highspy = None

    def _get_amat(self, T: int, eff: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (int(T), float(eff))
        cached = self._amat_cache.get(key)
        if cached is not None:
            return cached

        # Variables: [q_0..q_{T-1}, c_in_0..c_in_{T-1}, c_out_0..c_out_{T-1}]
        # Constraints (rows):
        # 0..T-2: q_{t+1} - q_t - eff*c_in_t + (1/eff)*c_out_t == 0
        # T-1:    q_0 - q_{T-1} - eff*c_in_{T-1} + (1/eff)*c_out_{T-1} == 0
        # T..2T-1: c_out_t - c_in_t <= u_t
        ncol = 3 * T
        nnz = 2 * ncol  # exactly 2 nonzeros per column
        start = (2 * np.arange(ncol + 1, dtype=np.int64)).copy()
        index = np.empty(nnz, dtype=np.int32)
        value = np.empty(nnz, dtype=np.float64)

        inv_eff = 1.0 / eff
        pos = 0

        # q columns
        cyc_row = T - 1
        for t in range(T):
            if t == 0:
                # row 0: -q0 ; cyclic row: +q0
                index[pos] = 0
                value[pos] = -1.0
                index[pos + 1] = cyc_row
                value[pos + 1] = 1.0
            elif t == T - 1:
                # row T-2: +q_{T-1} ; cyclic row: -q_{T-1}
                index[pos] = T - 2
                value[pos] = 1.0
                index[pos + 1] = cyc_row
                value[pos + 1] = -1.0
            else:
                # row t-1: +q_t ; row t: -q_t
                index[pos] = t - 1
                value[pos] = 1.0
                index[pos + 1] = t
                value[pos + 1] = -1.0
            pos += 2

        # c_in columns
        for t in range(T):
            dyn_row = cyc_row if t == T - 1 else t
            index[pos] = dyn_row
            value[pos] = -eff
            index[pos + 1] = T + t
            value[pos + 1] = -1.0
            pos += 2

        # c_out columns
        for t in range(T):
            dyn_row = cyc_row if t == T - 1 else t
            index[pos] = dyn_row
            value[pos] = inv_eff
            index[pos + 1] = T + t
            value[pos + 1] = 1.0
            pos += 2

        # Safety check (should always hold)
        if pos != nnz:
            raise RuntimeError("Internal error building sparse matrix.")

        self._amat_cache[key] = (start, index, value)
        return start, index, value

    def solve(self, problem: dict, **kwargs: Any) -> Any:
        # Some harness utilities may pass a JSON string; accept it.
        if isinstance(problem, str):
            import json

            problem = json.loads(problem)

        T = int(problem["T"])
        p = np.asarray(problem["p"], dtype=np.float64)
        u = np.asarray(problem["u"], dtype=np.float64)

        battery = problem["batteries"][0]
        Q = float(battery["Q"])
        C = float(battery["C"])
        D = float(battery["D"])
        eff = float(battery["efficiency"])

        # Try HiGHS first (fast path)
        if self._highs is not None and self._highspy is not None:
            highspy = self._highspy
            h = self._highs
            try:
                h.clear()

                lp = highspy.HighsLp()
                lp.num_col_ = 3 * T
                lp.num_row_ = 2 * T

                # Objective: minimize p @ (c_in - c_out)
                col_cost = np.empty(3 * T, dtype=np.float64)
                col_cost[:T] = 0.0
                col_cost[T : 2 * T] = p
                col_cost[2 * T :] = -p
                lp.col_cost_ = col_cost

                # Variable bounds
                col_lower = np.empty(3 * T, dtype=np.float64)
                col_upper = np.empty(3 * T, dtype=np.float64)
                col_lower[:T] = 0.0
                col_upper[:T] = Q
                col_lower[T : 2 * T] = 0.0
                col_upper[T : 2 * T] = C
                col_lower[2 * T :] = 0.0
                col_upper[2 * T :] = D
                lp.col_lower_ = col_lower
                lp.col_upper_ = col_upper

                # Row bounds
                row_lower = np.empty(2 * T, dtype=np.float64)
                row_upper = np.empty(2 * T, dtype=np.float64)
                row_lower[:T] = 0.0
                row_upper[:T] = 0.0
                row_lower[T:] = -self._inf
                row_upper[T:] = u
                lp.row_lower_ = row_lower
                lp.row_upper_ = row_upper

                start, index, value = self._get_amat(T, eff)
                lp.a_matrix_.start_ = start
                lp.a_matrix_.index_ = index
                lp.a_matrix_.value_ = value
                try:
                    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
                except Exception:
                    pass

                h.passModel(lp)
                h.run()

                try:
                    model_status = h.getModelStatus()
                    optimal = model_status in (
                        getattr(highspy.HighsModelStatus, "kOptimal", None),
                        getattr(highspy.HighsModelStatus, "kOptimal", None),
                    )
                except Exception:
                    # If status enum isn't accessible, assume success if solution exists.
                    optimal = True

                if not optimal:
                    raise RuntimeError("HiGHS did not report optimality.")

                sol = h.getSolution()
                x = np.asarray(sol.col_value, dtype=np.float64)

                q = x[:T]
                c_in = x[T : 2 * T]
                c_out = x[2 * T :]
                c_net = c_in - c_out

                cost_without = float(p @ u)
                cost_with = float(p @ (u + c_net))
                savings = cost_without - cost_with

                return {
                    "status": "optimal",
                    "optimal": True,
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
                    "savings_percent": float(100.0 * savings / cost_without) if cost_without != 0 else 0.0,
                }
            except Exception:
                # Fall back to CVXPY (slower but robust).
                pass

        # Fallback: CVXPY reference-style solve (should rarely execute).
        import cvxpy as cp

        q = cp.Variable(T)
        c_in = cp.Variable(T)
        c_out = cp.Variable(T)
        c = c_in - c_out

        constraints = [
            q >= 0,
            q <= Q,
            c_in >= 0,
            c_out >= 0,
            c_in <= C,
            c_out <= D,
            u + c >= 0,
        ]
        for t in range(T - 1):
            constraints.append(q[t + 1] == q[t] + eff * c_in[t] - (1.0 / eff) * c_out[t])
        constraints.append(q[0] == q[T - 1] + eff * c_in[T - 1] - (1.0 / eff) * c_out[T - 1])

        prob = cp.Problem(cp.Minimize(p @ c), constraints)
        prob.solve()

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return {"status": str(prob.status), "optimal": False}

        c_net = np.asarray(c_in.value - c_out.value, dtype=np.float64)
        cost_without = float(p @ u)
        cost_with = float(p @ (u + c_net))
        savings = cost_without - cost_with

        return {
            "status": str(prob.status),
            "optimal": True,
            "battery_results": [
                {
                    "q": np.asarray(q.value, dtype=np.float64).tolist(),
                    "c": c_net.tolist(),
                    "c_in": np.asarray(c_in.value, dtype=np.float64).tolist(),
                    "c_out": np.asarray(c_out.value, dtype=np.float64).tolist(),
                    "cost": cost_with,
                }
            ],
            "total_charging": c_net.tolist(),
            "cost_without_battery": cost_without,
            "cost_with_battery": cost_with,
            "savings": savings,
            "savings_percent": float(100.0 * savings / cost_without) if cost_without != 0 else 0.0,
        }