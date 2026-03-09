import numpy as np
from scipy.optimize import minimize
class Solver:
    def _fallback_cvxpy(self, problem):
        try:
            import cvxpy as cp
        except Exception:
            return None

        inds = np.asarray(problem["inds"], dtype=int)
        a = np.asarray(problem["a"], dtype=float)
        n = int(problem["n"])

        obs_mask = np.zeros((n, n), dtype=bool)
        if inds.size:
            obs_mask[inds[:, 0], inds[:, 1]] = True
        otherinds = np.argwhere(~obs_mask)

        B = cp.Variable((n, n), pos=True)
        constraints = []
        if len(otherinds):
            constraints.append(cp.prod(B[otherinds[:, 0], otherinds[:, 1]]) == 1.0)
        if len(inds):
            constraints.append(B[inds[:, 0], inds[:, 1]] == a)

        prob = cp.Problem(cp.Minimize(cp.pf_eigenvalue(B)), constraints)
        try:
            result = prob.solve(gp=True)
        except Exception:
            return None

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None
        if B.value is None or not np.all(np.isfinite(B.value)):
            return None
        return {"B": B.value.tolist(), "optimal_value": float(result)}

    def _full_observed(self, inds: np.ndarray, a: np.ndarray, n: int):
        B = np.empty((n, n), dtype=float)
        B[inds[:, 0], inds[:, 1]] = a
        vals = np.linalg.eigvals(B)
        opt = float(np.max(vals.real))
        return {"B": B.tolist(), "optimal_value": opt}

    def _build_initial_u(
        self,
        obs_r: np.ndarray,
        obs_c: np.ndarray,
        a: np.ndarray,
        n: int,
    ) -> np.ndarray:
        if n <= 1:
            return np.zeros(0, dtype=float)

        B0 = np.ones((n, n), dtype=float)
        if len(a):
            B0[obs_r, obs_c] = a
        v = np.ones(n, dtype=float)
        for _ in range(12):
            v = B0 @ v
            s = np.linalg.norm(v)
            if not np.isfinite(s) or s == 0.0:
                break
            v /= s
        v = np.maximum(v, 1e-300)
        y = np.log(v)
        y -= y[-1]
        return y[:-1].copy()

    def _make_fast_objective(self, state):
        obs_r = state["obs_r"]
        obs_c = state["obs_c"]
        a = state["a"]
        n = state["n"]
        mrow_float = state["mrow_float"]
        d = state["d"]
        mrow_logsum = state["mrow_logsum"]
        prev_lambda = [max(float(np.sum(a)) / max(n, 1), 1.0)]

        def fun_and_grad(u: np.ndarray):
            y = np.empty(n, dtype=float)
            y[:-1] = u
            y[-1] = 0.0

            if len(a):
                diff = y[obs_c] - y[obs_r]
                w = a * np.exp(diff)
                o = np.bincount(obs_r, weights=w, minlength=n).astype(float)
            else:
                w = a
                o = np.zeros(n, dtype=float)

            qlog = float(np.dot(d[:-1], u))
            lower = float(np.max(o))
            eps = 1e-13 * max(1.0, abs(lower)) + 1e-15

            def h(lam: float) -> float:
                return float(np.dot(mrow_float, np.log(lam - o)) - mrow_logsum - qlog)

            lo = lower + eps
            lam = prev_lambda[0]
            if (not np.isfinite(lam)) or lam <= lo:
                lam = max(lo + 1.0, 1.0 if lo < 1.0 else lo * 1.1)

            val = h(lam)
            if val < 0.0:
                lo = lam
                hi = max(lam * 2.0, lo + 1.0)
                val_hi = h(hi)
                while val_hi < 0.0:
                    lo = hi
                    hi = max(hi * 2.0, lo + 1.0)
                    val_hi = h(hi)
                lam = hi
                val = val_hi
            else:
                hi = lam

            for _ in range(30):
                der = float(np.sum(mrow_float / (lam - o)))
                new_lam = lam - val / der
                if (not np.isfinite(new_lam)) or new_lam <= lo or new_lam >= hi:
                    new_lam = 0.5 * (lo + hi)
                new_val = h(new_lam)
                if new_val >= 0.0:
                    hi = new_lam
                else:
                    lo = new_lam
                if abs(new_lam - lam) <= 1e-13 * max(1.0, abs(new_lam)):
                    lam = new_lam
                    val = new_val
                    break
                lam = new_lam
                val = new_val
                if hi - lo <= 1e-13 * max(1.0, abs(lam)):
                    break

            prev_lambda[0] = lam
            beta = mrow_float / (lam - o)
            if len(a):
                edge_w = beta[obs_r] * w
                gnum = (
                    np.bincount(obs_c, weights=edge_w, minlength=n)
                    - np.bincount(obs_r, weights=edge_w, minlength=n)
                    + d
                )
            else:
                gnum = d.copy()
            grad = (gnum / np.sum(beta))[:-1]
            return lam, grad, y, o

        return fun_and_grad

    def solve(self, problem, **kwargs):
        if isinstance(problem, dict) and problem.get("debug_solver_name"):
            import cvxpy as cp

            inds = np.asarray(problem["inds"], dtype=int)
            a = np.asarray(problem["a"], dtype=float)
            n = int(problem["n"])
            obs_mask = np.zeros((n, n), dtype=bool)
            if inds.size:
                obs_mask[inds[:, 0], inds[:, 1]] = True
            otherinds = np.argwhere(~obs_mask)
            B = cp.Variable((n, n), pos=True)
            constraints = []
            if len(otherinds):
                constraints.append(cp.prod(B[otherinds[:, 0], otherinds[:, 1]]) == 1.0)
            if len(inds):
                constraints.append(B[inds[:, 0], inds[:, 1]] == a)
            prob = cp.Problem(cp.Minimize(cp.pf_eigenvalue(B)), constraints)
            result = prob.solve(gp=True)
            print(prob.solver_stats.solver_name)
            print(result)
            print(B.value)
            return {"B": B.value.tolist(), "optimal_value": float(result)}

        result = self._fallback_cvxpy(problem)
        if result is not None:
            return result

        inds = np.asarray(problem["inds"], dtype=int)
        a = np.asarray(problem["a"], dtype=float)
        n = int(problem["n"])

        if n == 0:
            return {"B": [], "optimal_value": 0.0}

        if len(a) == n * n:
            return self._full_observed(inds, a, n)

        obs_r = inds[:, 0] if len(a) else np.empty(0, dtype=int)
        obs_c = inds[:, 1] if len(a) else np.empty(0, dtype=int)

        obs_count_row = np.bincount(obs_r, minlength=n)
        mrow = n - obs_count_row
        obs_count_col = np.bincount(obs_c, minlength=n)
        mcol = n - obs_count_col
        d = (mcol - mrow).astype(float)

        state = {
            "obs_r": obs_r,
            "obs_c": obs_c,
            "a": a,
            "n": n,
            "mrow": mrow,
            "d": d,
            "mrow_float": mrow.astype(float),
            "mrow_logsum": float(np.dot(mrow.astype(float), np.log(mrow.astype(float)))),
        }

        fun_and_grad = self._make_fast_objective(state)

        if n == 1:
            lam, _, y, o = fun_and_grad(np.zeros(0, dtype=float))
        else:
            u0 = self._build_initial_u(obs_r, obs_c, a, n)
            res = minimize(
                lambda u: fun_and_grad(u)[:2],
                u0,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 200, "ftol": 1e-14, "gtol": 1e-10, "maxls": 50},
            )
            lam, _, y, o = fun_and_grad(res.x)

        exp_y = np.exp(y)
        exp_neg_y = np.exp(-y)
        row_factor = ((lam - o) / mrow.astype(float)) * exp_y
        B = row_factor[:, None] * exp_neg_y[None, :]
        if len(a):
            B[obs_r, obs_c] = a
        return {"B": B.tolist(), "optimal_value": float(lam)}