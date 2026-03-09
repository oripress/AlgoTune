import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

class Solver:
    def __init__(self) -> None:
        self._template_cache = {}

    def _get_template(self, m: int, n: int):
        key = (m, n)
        template = self._template_cache.get(key)
        if template is not None:
            return template

        mn = m * n
        ad_idx = np.repeat(np.arange(m, dtype=np.int64), n)
        slot_idx = np.tile(np.arange(n, dtype=np.int64), m)

        indices = np.empty(3 * (mn + m), dtype=np.int64)
        indices[0 : 3 * mn : 3] = slot_idx
        indices[1 : 3 * mn : 3] = n + ad_idx
        indices[2 : 3 * mn : 3] = n + m + ad_idx

        s_rows = np.arange(m, dtype=np.int64)
        base = 3 * mn
        indices[base::3] = n + m + s_rows
        indices[base + 1 :: 3] = n + 2 * m + s_rows
        indices[base + 2 :: 3] = n + 3 * m

        indptr = np.arange(0, 3 * (mn + m) + 1, 3, dtype=np.int64)

        template = {
            "indices": indices,
            "indptr": indptr,
            "mn": mn,
        }
        self._template_cache[key] = template
        return template

    def _allocate_fillers(
        self,
        displays: np.ndarray,
        deficits: np.ndarray,
        leftover: np.ndarray,
    ) -> np.ndarray:
        m, n = displays.shape
        t = 0
        for i in range(m):
            need = deficits[i]
            while need > 1e-12:
                while t < n and leftover[t] <= 1e-12:
                    t += 1
                if t >= n:
                    break
                add = need if need < leftover[t] else leftover[t]
                displays[i, t] += add
                leftover[t] -= add
                need -= add
        return displays

    def solve(self, problem: dict, **kwargs) -> dict:
        try:
            P = np.asarray(problem["P"], dtype=float)
            R = np.asarray(problem["R"], dtype=float)
            B = np.asarray(problem["B"], dtype=float)
            c = np.asarray(problem["c"], dtype=float)
            T = np.asarray(problem["T"], dtype=float)

            if P.ndim != 2:
                return {"status": "error", "optimal": False, "error": "P must be a 2D matrix"}

            m, n = P.shape
            if R.shape != (m,) or B.shape != (m,) or c.shape != (m,) or T.shape != (n,):
                return {"status": "error", "optimal": False, "error": "Invalid input dimensions"}

            total_capacity = float(np.sum(T))
            if total_capacity + 1e-9 < float(np.sum(c)):
                return {"status": "infeasible", "optimal": False}

            w = (P * R[:, None]).astype(float, copy=False)
            w_flat = w.ravel()
            template = self._get_template(m, n)
            mn = template["mn"]

            data = np.empty(3 * (mn + m), dtype=float)
            data[0 : 3 * mn : 3] = 1.0
            data[1 : 3 * mn : 3] = w_flat
            data[2 : 3 * mn : 3] = 1.0
            tail = 3 * mn
            data[tail::3] = -1.0
            data[tail + 1 :: 3] = -1.0
            data[tail + 2 :: 3] = 1.0

            a_ub = csc_matrix(
                (data, template["indices"], template["indptr"]),
                shape=(n + 3 * m + 1, mn + m),
            )

            b_ub = np.empty(n + 3 * m + 1, dtype=float)
            b_ub[:n] = T
            b_ub[n : n + m] = B
            b_ub[n + m : n + 2 * m] = 0.0
            b_ub[n + 2 * m : n + 3 * m] = -c
            b_ub[n + 3 * m] = total_capacity

            c_obj = np.empty(mn + m, dtype=float)
            c_obj[:mn] = -w_flat
            c_obj[mn:] = 0.0

            res = linprog(
                c=c_obj,
                A_ub=a_ub,
                b_ub=b_ub,
                bounds=(0, None),
                method="highs",
            )

            if not res.success or res.x is None:
                status = str(res.status) if getattr(res, "status", None) is not None else "solver_error"
                message = res.message if getattr(res, "message", None) else status
                low = message.lower()
                if "infeasible" in low:
                    return {"status": "infeasible", "optimal": False}
                return {"status": message, "optimal": False}

            x = res.x[:mn].reshape(m, n)
            if x.min() < 0.0:
                x = x.copy()
                x[x < 0.0] = 0.0

            row_x = x.sum(axis=1)
            row_x = x.sum(axis=1)
            deficits = np.maximum(c - row_x, 0.0)
            if float(deficits.max(initial=0.0)) <= 1e-12:
                displays = x
            else:
                col_x = x.sum(axis=0)
                leftover = np.maximum(T - col_x, 0.0)
                displays = self._allocate_fillers(x.copy(), deficits, leftover)
            clicks = np.sum(P * displays, axis=1)
            revenue = np.minimum(R * clicks, B)
            total_revenue = float(np.sum(revenue))

            return {
                "status": "optimal",
                "optimal": True,
                "displays": displays,
                "clicks": clicks,
                "revenue_per_ad": revenue,
                "total_revenue": total_revenue,
            }
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}