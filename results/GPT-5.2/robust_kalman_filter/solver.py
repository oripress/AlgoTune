from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cvxpy as cp
except Exception:  # pragma: no cover
    cp = None  # type: ignore[assignment]

class _CvxCache:
    __slots__ = ("prob", "A", "B", "C", "y", "x0", "tau", "M", "x", "w", "v")

    def __init__(
        self,
        prob: "cp.Problem",
        A: "cp.Parameter",
        B: "cp.Parameter",
        C: "cp.Parameter",
        y: "cp.Parameter",
        x0: "cp.Parameter",
        tau: "cp.Parameter",
        M: "cp.Parameter",
        x: "cp.Variable",
        w: "cp.Variable",
        v: "cp.Variable",
    ) -> None:
        self.prob = prob
        self.A = A
        self.B = B
        self.C = C
        self.y = y
        self.x0 = x0
        self.tau = tau
        self.M = M
        self.x = x
        self.w = w
        self.v = v

def _build_problem(N: int, n: int, m: int, p: int) -> _CvxCache:
    # Parameters
    A = cp.Parameter((n, n), name="A")
    B = cp.Parameter((n, p), name="B")
    C = cp.Parameter((m, n), name="C")
    y = cp.Parameter((N, m), name="y")
    x0 = cp.Parameter((n,), name="x0")
    tau = cp.Parameter(nonneg=True, name="tau")
    M = cp.Parameter(nonneg=True, name="M")

    # Variables
    x = cp.Variable((N + 1, n), name="x")
    w = cp.Variable((N, p), name="w")
    v = cp.Variable((N, m), name="v")

    # Vectorized constraints (only 3 constraints vs O(N) in the reference)
    cons = [
        x[0, :] == x0,
        x[1:, :] == x[:-1, :] @ A.T + w @ B.T,
        y == x[:-1, :] @ C.T + v,
    ]

    # Vectorized robust objective
    meas = cp.sum(cp.huber(cp.norm(v, axis=1), M))
    obj = cp.sum_squares(w) + tau * meas

    prob = cp.Problem(cp.Minimize(obj), cons)
    return _CvxCache(prob=prob, A=A, B=B, C=C, y=y, x0=x0, tau=tau, M=M, x=x, w=w, v=v)

class Solver:
    def __init__(self) -> None:
        # key: (N,n,m,p)
        self._cache: dict[tuple[int, int, int, int], _CvxCache] = {}

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        if cp is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        C = np.asarray(problem["C"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        x0 = np.asarray(problem["x_initial"], dtype=float)
        tau = float(problem["tau"])
        M = float(problem["M"])

        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        key = (N, n, m, p)
        cache = self._cache.get(key)
        if cache is None:
            cache = _build_problem(N, n, m, p)
            self._cache[key] = cache

        cache.A.value = A
        cache.B.value = B
        cache.C.value = C
        cache.y.value = y
        cache.x0.value = x0
        cache.tau.value = tau
        cache.M.value = M

        try:
            cache.prob.solve(
                solver=cp.ECOS,
                warm_start=True,
                feastol=1e-6,
                abstol=1e-6,
                reltol=1e-6,
                max_iters=50,
                verbose=False,
            )
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        if cache.prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or cache.x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        return {
            "x_hat": cache.x.value.tolist(),
            "w_hat": cache.w.value.tolist(),
            "v_hat": cache.v.value.tolist(),
        }