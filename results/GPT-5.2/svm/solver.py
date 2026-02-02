from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import cvxpy as cp
except Exception:  # pragma: no cover
    cp = None  # type: ignore[assignment]

try:
    import osqp  # type: ignore
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover
    osqp = None  # type: ignore[assignment]
    sp = None  # type: ignore[assignment]

class _SVMModel:
    __slots__ = ("prob", "Xy_param", "y_param", "C_param", "beta", "beta0", "Xy_buf")

    def __init__(
        self,
        prob: Any,
        Xy_param: Any,
        y_param: Any,
        C_param: Any,
        beta: Any,
        beta0: Any,
        Xy_buf: np.ndarray,
    ) -> None:
        self.prob = prob
        self.Xy_param = Xy_param
        self.y_param = y_param
        self.C_param = C_param
        self.beta = beta
        self.beta0 = beta0
        self.Xy_buf = Xy_buf

class _OSQPModel:
    __slots__ = ("solver", "n", "p", "m", "A_data", "q", "Xy_buf_f")

    def __init__(
        self,
        solver: Any,
        n: int,
        p: int,
        m: int,
        A_data: np.ndarray,
        q: np.ndarray,
        Xy_buf_f: np.ndarray,
    ) -> None:
        self.solver = solver
        self.n = n
        self.p = p
        self.m = m
        self.A_data = A_data
        self.q = q
        self.Xy_buf_f = Xy_buf_f

class Solver:
    """
    Fast primal soft-margin linear SVM.

    Uses a direct OSQP QP solve (cached per (n,p) within this Solver instance)
    for speed, with a cached CVXPY fallback to match the reference behavior.
    """

    def __init__(self) -> None:
        self._cvx_cache: dict[tuple[int, int], _SVMModel] = {}
        self._osqp_cache: dict[tuple[int, int], _OSQPModel] = {}

        self._has_osqp = osqp is not None and sp is not None

        self._cvx_solver: Optional[str] = None
        if cp is not None:
            try:
                if "OSQP" in set(cp.installed_solvers()):
                    self._cvx_solver = "OSQP"
            except Exception:
                self._cvx_solver = None

    # -------------------------
    # CVXPY fallback (cached)
    # -------------------------
    def _get_cvx_model(self, n: int, p: int) -> _SVMModel:
        key = (n, p)
        model = self._cvx_cache.get(key)
        if model is not None:
            return model
        if cp is None:
            raise RuntimeError("cvxpy is required for the fallback path.")

        # DPP-friendly parameters:
        # Xy = diag(y) @ X  stored as elementwise product X * y[:, None]
        Xy_param = cp.Parameter((n, p))
        y_param = cp.Parameter(n)
        C_param = cp.Parameter(nonneg=True)

        beta = cp.Variable(p)
        beta0 = cp.Variable()
        xi = cp.Variable(n)

        constraints = [
            xi >= 0,
            Xy_param @ beta + y_param * beta0 >= 1 - xi,
        ]
        objective = cp.Minimize(0.5 * cp.sum_squares(beta) + C_param * cp.sum(xi))
        prob = cp.Problem(objective, constraints)

        model = _SVMModel(
            prob=prob,
            Xy_param=Xy_param,
            y_param=y_param,
            C_param=C_param,
            beta=beta,
            beta0=beta0,
            Xy_buf=np.empty((n, p), dtype=np.float64),
        )
        self._cvx_cache[key] = model
        return model
    def _solve_via_cvxpy(self, X: np.ndarray, y: np.ndarray, C: float) -> Any:
        if cp is None:
            return None
        n, p = X.shape
        model = self._get_cvx_model(n, p)

        np.multiply(X, y[:, None], out=model.Xy_buf)
        model.Xy_param.value = model.Xy_buf
        model.y_param.value = y
        model.C_param.value = C

        try:
            if self._cvx_solver is None:
                optimal_value = model.prob.solve(warm_start=True)
            else:
                optimal_value = model.prob.solve(
                    solver=self._cvx_solver, warm_start=True
                )
        except Exception:
            return None

        status = getattr(model.prob, "status", None)
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None

        beta_val = model.beta.value
        beta0_val = model.beta0.value
        if beta_val is None or beta0_val is None:
            return None

        beta_val = np.asarray(beta_val, dtype=np.float64).reshape(-1)
        beta0_val_f = float(beta0_val)

        pred = X @ beta_val + beta0_val_f
        missclass = float(np.mean((pred * y) < 0))

        return {
            "beta0": beta0_val_f,
            "beta": beta_val.tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": missclass,
        }

    # -------------------------
    # Direct OSQP path (cached)
    # -------------------------
    def _get_osqp_model(self, n: int, p: int) -> _OSQPModel:
        key = (n, p)
        model = self._osqp_cache.get(key)
        if model is not None:
            return model
        if not self._has_osqp:
            raise RuntimeError("osqp/scipy not available.")

        # Variables: x = [beta (p), beta0 (1), xi (n)] => m = p+1+n
        m = p + 1 + n

        # Objective: 0.5*||beta||^2 + C*sum(xi)
        diag = np.zeros(m, dtype=np.float64)
        diag[:p] = 1.0
        P = sp.diags(diag, 0, shape=(m, m), format="csc")
        q = np.zeros(m, dtype=np.float64)

        # Constraints (2n):
        # 1) margin: (diag(y)X)beta + y*beta0 + xi >= 1
        # 2) xi >= 0
        #
        # A has fixed sparsity pattern:
        # - beta cols: n entries (rows 0..n-1)
        # - beta0 col: n entries (rows 0..n-1)
        # - xi cols: 2 entries (rows i and n+i)
        nnz = n * (p + 1) + 2 * n
        indptr = np.empty(m + 1, dtype=np.int64)

        # First p+1 columns: n entries each
        indptr[: p + 2] = np.arange(0, (p + 2) * n, n, dtype=np.int64)
        base = (p + 1) * n
        # Xi columns: 2 entries each
        indptr[p + 2 :] = base + 2 * np.arange(1, n + 1, dtype=np.int64)

        idx_beta = np.tile(np.arange(n, dtype=np.int64), p + 1)
        idx_xi = np.empty(2 * n, dtype=np.int64)
        idx_xi[0::2] = np.arange(n, dtype=np.int64)
        idx_xi[1::2] = n + np.arange(n, dtype=np.int64)
        indices = np.concatenate((idx_beta, idx_xi), axis=0)

        A_data = np.ones(nnz, dtype=np.float64)
        A = sp.csc_matrix((A_data, indices, indptr), shape=(2 * n, m))

        l = np.empty(2 * n, dtype=np.float64)
        l[:n] = 1.0
        l[n:] = 0.0
        u = np.full(2 * n, np.inf, dtype=np.float64)

        solver = osqp.OSQP()
        solver.setup(
            P=P,
            q=q,
            A=A,
            l=l,
            u=u,
            verbose=False,
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=10000,
            polish=True,
            adaptive_rho=True,
            warm_start=False,
        )

        model = _OSQPModel(
            solver=solver,
            n=n,
            p=p,
            m=m,
            A_data=A_data,
            q=q,
            Xy_buf_f=np.empty((n, p), dtype=np.float64, order="F"),
        )
        self._osqp_cache[key] = model
        return model

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64).reshape(-1)
        C = float(problem["C"])
        n, p = X.shape

        if self._has_osqp:
            try:
                model = self._get_osqp_model(n, p)

                # Update q: only xi part depends on C.
                model.q[p + 1 :] = C

                # Update A data: [vec(Xy) in CSC order, y, ones(2n)].
                np.multiply(X, y[:, None], out=model.Xy_buf_f)
                k = n * p
                model.A_data[:k] = model.Xy_buf_f.ravel(order="F")
                model.A_data[k : k + n] = y
                # remaining 2n entries are constant ones

                model.solver.update(q=model.q, Ax=model.A_data)
                res = model.solver.solve()

                status = getattr(res.info, "status", "")
                if not (status.startswith("solved") or status.startswith("optimal")):
                    return self._solve_via_cvxpy(X, y, C)

                x = np.asarray(res.x, dtype=np.float64)
                beta = x[:p]
                beta0 = float(x[p])

                pred = X @ beta + beta0
                missclass = float(np.mean((pred * y) < 0))

                optimal_value = float(getattr(res.info, "obj_val", np.nan))
                if not np.isfinite(optimal_value):
                    optimal_value = float(0.5 * beta.dot(beta) + C * np.sum(x[p + 1 :]))

                return {
                    "beta0": beta0,
                    "beta": beta.tolist(),
                    "optimal_value": optimal_value,
                    "missclass_error": missclass,
                }
            except Exception:
                return self._solve_via_cvxpy(X, y, C)

        return self._solve_via_cvxpy(X, y, C)