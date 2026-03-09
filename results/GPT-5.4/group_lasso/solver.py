from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np

@dataclass
class _CachedProblem:
    X_param: cp.Parameter
    y_param: cp.Parameter
    lba_param: cp.Parameter
    beta_var: cp.Variable
    beta0_var: cp.Variable
    problem: cp.Problem
    inverse: np.ndarray

class Solver:
    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, tuple[Any, ...]], _CachedProblem] = {}

    def _build_cached_problem(
        self,
        n: int,
        p: int,
        gl: np.ndarray,
    ) -> _CachedProblem:
        key = (n, p, tuple(np.asarray(gl).tolist()))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        ulabels, inverseinds, pjs = np.unique(
            gl[:, None], return_inverse=True, return_counts=True
        )
        m = int(ulabels.shape[0])

        group_idx = np.zeros((p, m), dtype=bool)
        group_idx[np.arange(p), inverseinds.flatten()] = True
        not_group_idx = np.logical_not(group_idx)
        sqr_group_sizes = np.sqrt(pjs.astype(float))

        X_param = cp.Parameter((n, p))
        y_param = cp.Parameter((n, 1))
        lba_param = cp.Parameter(nonneg=True)

        beta = cp.Variable((p, m))
        beta0 = cp.Variable()

        xb = cp.sum(X_param @ beta, axis=1, keepdims=True) + beta0
        logreg = -cp.sum(cp.multiply(y_param, xb)) + cp.sum(cp.logistic(cp.reshape(xb, (n,))))
        grouplasso = lba_param * cp.sum(
            cp.multiply(cp.norm(beta, 2, axis=0), sqr_group_sizes)
        )
        objective = cp.Minimize(logreg + grouplasso)
        constraints = [beta[not_group_idx] == 0]
        problem = cp.Problem(objective, constraints)

        cached = _CachedProblem(
            X_param=X_param,
            y_param=y_param,
            lba_param=lba_param,
            beta_var=beta,
            beta0_var=beta0,
            problem=problem,
            inverse=inverseinds.flatten(),
        )
        self._cache[key] = cached
        return cached

    def solve(self, problem, **kwargs) -> Any:
        X = np.asarray(problem["X"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        gl = np.asarray(problem["gl"])
        lba = float(problem["lba"])

        n = int(X.shape[0])
        p = int(X.shape[1] - 1)

        if p == 0:
            beta0 = cp.Variable()
            y_col = y[:, None]
            xb = beta0
            obj = cp.Minimize(-cp.sum(cp.multiply(y_col, xb)) + n * cp.logistic(beta0))
            prob = cp.Problem(obj)
            try:
                result = prob.solve(solver=cp.ECOS, warm_start=True, ignore_dpp=True)
            except cp.SolverError:
                return None
            except Exception:
                return None
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return None
            if beta0.value is None:
                return None
            return {"beta0": float(beta0.value), "beta": [], "optimal_value": float(result)}

        cached = self._build_cached_problem(n, p, gl)
        cached.X_param.value = X[:, 1:]
        cached.y_param.value = y[:, None]
        cached.lba_param.value = lba

        try:
            result = cached.problem.solve(warm_start=True, ignore_dpp=True)
        except cp.SolverError:
            try:
                result = cached.problem.solve(solver=cp.ECOS, warm_start=True, ignore_dpp=True)
            except cp.SolverError:
                return None
            except Exception:
                return None
        except Exception:
            return None

        if cached.problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None

        beta_val = cached.beta_var.value
        beta0_val = cached.beta0_var.value
        if beta_val is None or beta0_val is None:
            return None

        beta = beta_val[np.arange(p), cached.inverse]
        return {
            "beta0": float(beta0_val),
            "beta": beta.tolist(),
            "optimal_value": float(result),
        }