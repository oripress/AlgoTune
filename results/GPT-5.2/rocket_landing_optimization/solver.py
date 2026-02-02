from typing import Any, Dict

import numpy as np

# Import CVXPY at module import time to avoid solve()-time overhead.
import cvxpy as cp

class _CachedProblem:
    __slots__ = (
        "K",
        "p0",
        "v0",
        "p_target",
        "alpha",
        "grav",
        "h2",
        "fmax",
        "V",
        "P",
        "F",
        "prob",
        "last_F",
    )

    def __init__(
        self,
        K: int,
        p0: cp.Parameter,
        v0: cp.Parameter,
        p_target: cp.Parameter,
        alpha: cp.Parameter,
        grav: cp.Parameter,
        h2: cp.Parameter,
        fmax: cp.Parameter,
        V: cp.Variable,
        P: cp.Variable,
        F: cp.Variable,
        prob: cp.Problem,
    ) -> None:
        self.K = K
        self.p0 = p0
        self.v0 = v0
        self.p_target = p_target
        self.alpha = alpha
        self.grav = grav
        self.h2 = h2
        self.fmax = fmax
        self.V = V
        self.P = P
        self.F = F
        self.prob = prob
        self.last_F = None

class Solver:
    """
    Fast SOCP solver using a cached, DPP-compliant CVXPY model (per K).

    Uses Parameters for p0, v0, target, and scalar constants so canonicalization
    can be reused across instances with the same K. Solves with ECOS warm-start.
    """

    def __init__(self) -> None:
        self._cache: Dict[int, _CachedProblem] = {}

    @staticmethod
    def _as3(x: Any) -> np.ndarray:
        a = np.asarray(x, dtype=np.float64).reshape(3)
        return a

    def _get_cached(self, K: int) -> _CachedProblem:
        cached = self._cache.get(K)
        if cached is not None:
            return cached

        # Parameters (affine usage -> DPP)
        p0 = cp.Parameter(3, name="p0")
        v0 = cp.Parameter(3, name="v0")
        p_target = cp.Parameter(3, name="p_target")

        alpha = cp.Parameter(nonneg=True, name="alpha")  # h/m
        grav = cp.Parameter(nonneg=True, name="grav")  # h*g
        h2 = cp.Parameter(nonneg=True, name="h2")  # h/2
        fmax = cp.Parameter(nonneg=True, name="fmax")  # F_max

        V = cp.Variable((K + 1, 3), name="V")
        P = cp.Variable((K + 1, 3), name="P")
        F = cp.Variable((K, 3), name="F")

        constraints = [
            V[0, :] == v0,
            P[0, :] == p0,
            V[K, :] == 0.0,
            P[K, :] == p_target,
            P[:, 2] >= 0.0,
            V[1:, :2] == V[:-1, :2] + alpha * F[:, :2],
            V[1:, 2] == V[:-1, 2] + alpha * F[:, 2] - grav,
            P[1:, :] == P[:-1, :] + h2 * (V[:-1, :] + V[1:, :]),
            cp.norm(F, 2, axis=1) <= fmax,
        ]

        # gamma doesn't change argmin; compute fuel separately for output.
        objective = cp.Minimize(cp.sum(cp.norm(F, 2, axis=1)))
        prob = cp.Problem(objective, constraints)

        cached = _CachedProblem(
            K,
            p0,
            v0,
            p_target,
            alpha,
            grav,
            h2,
            fmax,
            V,
            P,
            F,
            prob,
        )
        self._cache[K] = cached
        return cached

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        K = int(problem["K"])
        p0 = self._as3(problem["p0"])
        v0 = self._as3(problem["v0"])
        p_target = self._as3(problem["p_target"])

        g = float(problem["g"])
        m = float(problem["m"])
        h = float(problem["h"])
        F_max = float(problem["F_max"])
        gamma = float(problem["gamma"])

        cached = self._get_cached(K)

        cached.p0.value = p0
        cached.v0.value = v0
        cached.p_target.value = p_target
        cached.alpha.value = h / m
        cached.grav.value = h * g
        cached.h2.value = 0.5 * h
        cached.fmax.value = F_max

        if cached.last_F is not None and cached.last_F.shape == (K, 3):
            cached.F.value = cached.last_F

        try:
            cached.prob.solve(
                solver=cp.ECOS,
                warm_start=True,
                verbose=False,
                max_iters=200,
            )
        except Exception:
            return {"position": [], "velocity": [], "thrust": []}

        if cached.prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return {"position": [], "velocity": [], "thrust": []}

        P_val = cached.P.value
        V_val = cached.V.value
        F_val = cached.F.value
        if P_val is None or V_val is None or F_val is None:
            return {"position": [], "velocity": [], "thrust": []}

        cached.last_F = np.asarray(F_val, dtype=np.float64, order="C").copy()
        fuel = gamma * float(np.sum(np.linalg.norm(F_val, axis=1)))

        # Return numpy arrays (validator accepts np.array(...) conversion)
        return {
            "position": P_val,
            "velocity": V_val,
            "thrust": F_val,
            "fuel_consumption": fuel,
        }