from __future__ import annotations

from typing import Any

import numpy as np

# CVXPY is required for exact matching with the reference (which also uses CVXPY).
import cvxpy as cp

class _Model:
    __slots__ = ("n", "B", "a_param", "prob")

    def __init__(self, n: int, inds: np.ndarray):
        self.n = n
        inds = np.asarray(inds, dtype=np.int64)
        obs_r = inds[:, 0]
        obs_c = inds[:, 1]

        obs_mask = np.zeros((n, n), dtype=bool)
        obs_mask[obs_r, obs_c] = True
        unobs_r, unobs_c = np.nonzero(~obs_mask)

        B = cp.Variable((n, n), pos=True)
        a_param = cp.Parameter(shape=(inds.shape[0],), pos=True)

        constraints = [
            B[obs_r, obs_c] == a_param,
        ]
        # Product constraint on all unobserved entries.
        if unobs_r.size:
            constraints.append(cp.prod(B[unobs_r, unobs_c]) == 1.0)

        objective = cp.Minimize(cp.pf_eigenvalue(B))
        prob = cp.Problem(objective, constraints)

        self.B = B
        self.a_param = a_param
        self.prob = prob

class Solver:
    def __init__(self) -> None:
        # Cache models by (n, m, inds_bytes). Order-sensitive but cheap.
        self._models: dict[tuple[int, int, bytes], _Model] = {}

    def solve(self, problem, **kwargs) -> Any:
        inds = np.asarray(problem["inds"], dtype=np.int64)
        a = np.asarray(problem["a"], dtype=np.float64)
        n = int(problem["n"])

        if inds.ndim != 2 or inds.shape[1] != 2:
            return None
        if a.shape[0] != inds.shape[0]:
            return None

        # Ensure contiguous for stable bytes key.
        inds_c = np.ascontiguousarray(inds)
        key = (n, int(inds_c.shape[0]), inds_c.tobytes())

        model = self._models.get(key)
        if model is None:
            model = _Model(n, inds_c)
            self._models[key] = model

        # CVXPY expects strictly positive for gp=True; clip tiny values.
        model.a_param.value = np.maximum(a, 1e-300)

        try:
            # Match reference as closely as possible: use gp=True, default solver selection.
            result = model.prob.solve(gp=True, warm_start=True)
        except Exception:
            return None

        Bv = model.B.value
        if Bv is None or not np.all(np.isfinite(Bv)):
            return None

        return {"B": Bv.tolist(), "optimal_value": float(result)}