import itertools
import math
from functools import lru_cache
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt
import scipy.sparse as sp

try:
    import osqp  # type: ignore

    _HAS_OSQP = True
except Exception:  # pragma: no cover
    osqp = None
    _HAS_OSQP = False

try:
    import cvxpy as cp  # type: ignore

    _HAS_CVXPY = True
except Exception:  # pragma: no cover
    cp = None
    _HAS_CVXPY = False

def _get_P(problem: dict[str, Any]) -> np.ndarray:
    if "P" in problem:
        return np.asarray(problem["P"], dtype=float)
    return np.asarray(problem["Q"], dtype=float)

def _as_vec(value: Any, size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if size is not None and arr.size == 0:
        return np.zeros(size, dtype=float)
    return arr

def _as_mat(value: Any, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.zeros((0, n), dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr

def _objective(P: np.ndarray, q: np.ndarray, x: np.ndarray) -> float:
    return float(0.5 * x @ P @ x + q @ x)

def _solve_linear(P: np.ndarray, q: np.ndarray) -> np.ndarray:
    try:
        return -sla.solve(P, q, assume_a="sym")
    except Exception:
        return -sla.lstsq(P, q)[0]

def _solve_kkt_full(
    P: np.ndarray,
    q: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = P.shape[0]
    p = A.shape[0]
    if p == 0:
        return _solve_linear(P, q), np.zeros(0, dtype=float)
    K = np.empty((n + p, n + p), dtype=float)
    K[:n, :n] = P
    K[:n, n:] = A.T
    K[n:, :n] = A
    K[n:, n:] = 0.0
    rhs = np.empty(n + p, dtype=float)
    rhs[:n] = -q
    rhs[n:] = b
    try:
        sol = sla.solve(K, rhs, assume_a="sym")
    except Exception:
        sol = sla.lstsq(K, rhs)[0]
    return sol[:n], sol[n:]

def _solve_kkt(P: np.ndarray, q: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _solve_kkt_full(P, q, A, b)[0]

def _feasible(
    x: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    atol: float = 2e-6,
) -> bool:
    if G.shape[0] and np.any(G @ x - h > atol):
        return False
    if A.shape[0] and not np.allclose(A @ x, b, atol=atol):
        return False
    return True

def _active_set_polish(
    x: np.ndarray,
    P: np.ndarray,
    q: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    m = G.shape[0]
    if m == 0:
        return x
    slack = h - G @ x
    tol = max(1e-7, 1e-7 * (1.0 + float(np.max(np.abs(h))) if h.size else 1.0))
    active = slack <= tol
    if not np.any(active):
        return x
    A2 = G[active]
    b2 = h[active]
    if A.shape[0]:
        Aeq = np.vstack((A, A2))
        beq = np.concatenate((b, b2))
    else:
        Aeq = A2
        beq = b2
    x2 = _solve_kkt(P, q, Aeq, beq)
    if _feasible(x2, G, h, A, b):
        obj1 = _objective(P, q, x)
        obj2 = _objective(P, q, x2)
        if obj2 <= obj1 + 1e-9 * (1.0 + abs(obj1)):
            return x2
    return x

_OSQP_CACHE: dict[tuple[int, int], Any] = {}

@lru_cache(maxsize=None)
def _triu_pack_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.triu_indices(n)
    order = np.lexsort((rows, cols))
    return rows[order], cols[order]

def _pack_triu(P: np.ndarray) -> np.ndarray:
    rows, cols = _triu_pack_indices(P.shape[0])
    return np.asarray(P[rows, cols], dtype=float)

def _pack_dense_cols(M: np.ndarray) -> np.ndarray:
    return np.asarray(M.T, dtype=float).reshape(-1)

def _get_osqp_solver(n: int, m: int):
    key = (n, m)
    cached = _OSQP_CACHE.get(key)
    if cached is not None:
        return cached

    rows, _ = _triu_pack_indices(n)
    p_indptr = np.empty(n + 1, dtype=np.int32)
    p_indptr[0] = 0
    np.cumsum(np.arange(1, n + 1, dtype=np.int32), out=p_indptr[1:])
    P0 = sp.csc_matrix(
        (np.zeros(rows.size, dtype=float), rows.astype(np.int32), p_indptr),
        shape=(n, n),
    )

    if m:
        a_indices = np.tile(np.arange(m, dtype=np.int32), n)
        a_indptr = np.arange(n + 1, dtype=np.int32) * m
        A0 = sp.csc_matrix(
            (np.zeros(m * n, dtype=float), a_indices, a_indptr),
            shape=(m, n),
        )
    else:
        A0 = sp.csc_matrix((0, n), dtype=float)

    solver = osqp.OSQP()
    solver.setup(
        P=P0,
        q=np.zeros(n, dtype=float),
        A=A0,
        l=np.zeros(m, dtype=float),
        u=np.zeros(m, dtype=float),
        verbose=False,
        eps_abs=1e-8,
        eps_rel=1e-8,
        polish=True,
        adaptive_rho=True,
        scaling=10,
        warm_start=True,
        max_iter=100000,
        check_termination=25,
    )
    _OSQP_CACHE[key] = solver
    return solver

def _solve_active_set_small(
    P: np.ndarray,
    q: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
) -> np.ndarray | None:
    n = P.shape[0]
    m = G.shape[0]
    p = A.shape[0]
    if m == 0 or n > 8 or m > 18:
        return None

    if p:
        rank_a = int(np.linalg.matrix_rank(A))
    else:
        rank_a = 0
    max_active = min(m, max(0, n - rank_a))

    total = 0
    for k in range(max_active + 1):
        total += math.comb(m, k)
        if total > 5000:
            return None

    best_x = None
    best_obj = math.inf

    x = _solve_kkt(P, q, A, b)
    if np.all(np.isfinite(x)) and _feasible(x, G, h, A, b):
        best_x = x
        best_obj = _objective(P, q, x)

    for k in range(1, max_active + 1):
        for combo in itertools.combinations(range(m), k):
            idx = np.fromiter(combo, dtype=np.int32, count=k)
            if p:
                Aeq = np.empty((p + k, n), dtype=float)
                Aeq[:p] = A
                Aeq[p:] = G[idx]
                beq = np.empty(p + k, dtype=float)
                beq[:p] = b
                beq[p:] = h[idx]
            else:
                Aeq = G[idx]
                beq = h[idx]
            x = _solve_kkt(P, q, Aeq, beq)
            if not np.all(np.isfinite(x)) or not _feasible(x, G, h, A, b):
                continue
            obj = _objective(P, q, x)
            if obj < best_obj:
                best_obj = obj
                best_x = x

    if best_x is None:
        return None
    return _active_set_polish(best_x, P, q, G, h, A, b)

def _solve_osqp(
    P: np.ndarray,
    q: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
) -> np.ndarray | None:
    if not _HAS_OSQP:
        return None
    n = P.shape[0]
    if G.shape[0]:
        if A.shape[0]:
            AA = np.vstack((G, A))
            l = np.concatenate((np.full(G.shape[0], -np.inf, dtype=float), b))
            u = np.concatenate((h, b))
        else:
            AA = G
            l = np.full(G.shape[0], -np.inf, dtype=float)
            u = h
    else:
        AA = A
        l = b
        u = b

    try:
        solver = _get_osqp_solver(n, AA.shape[0])
        solver.update(Px=_pack_triu(P), Ax=_pack_dense_cols(AA), q=q, l=l, u=u)
        res = solver.solve()
    except Exception:
        return None

    status = getattr(getattr(res, "info", None), "status", "")
    x = getattr(res, "x", None)
    if x is None or "solved" not in str(status).lower():
        return None
    x = np.asarray(x, dtype=float)
    if not _feasible(x, G, h, A, b, atol=5e-5):
        return None
    return _active_set_polish(x, P, q, G, h, A, b)

def _initial_guess(
    P: np.ndarray,
    q: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    if A.shape[0]:
        x0 = _solve_kkt(P + 1e-9 * np.eye(P.shape[0]), q, A, b)
        if np.all(np.isfinite(x0)):
            return x0
        try:
            return sla.lstsq(A, b)[0]
        except Exception:
            pass
    return np.zeros(P.shape[0], dtype=float)

def _solve_scipy(
    P: np.ndarray,
    q: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
) -> np.ndarray | None:
    x0 = _initial_guess(P, q, G, h, A, b)

    def fun(x: np.ndarray) -> float:
        return 0.5 * x @ P @ x + q @ x

    def jac(x: np.ndarray) -> np.ndarray:
        return P @ x + q

    constraints: list[Any] = []
    if G.shape[0]:
        constraints.append(sopt.LinearConstraint(G, -np.inf * np.ones(G.shape[0]), h))
    if A.shape[0]:
        constraints.append(sopt.LinearConstraint(A, b, b))

    try:
        res = sopt.minimize(
            fun,
            x0,
            method="trust-constr",
            jac=jac,
            hess=lambda _x: P,
            constraints=constraints,
            options={
                "verbose": 0,
                "gtol": 1e-10,
                "xtol": 1e-12,
                "barrier_tol": 1e-12,
                "maxiter": 500,
            },
        )
        x = np.asarray(res.x, dtype=float)
        if res.success and _feasible(x, G, h, A, b):
            return _active_set_polish(x, P, q, G, h, A, b)
    except Exception:
        pass

    if G.shape[0]:
        cons = [{"type": "ineq", "fun": lambda x, G=G, h=h: h - G @ x}]
    else:
        cons = []
    if A.shape[0]:
        cons.append({"type": "eq", "fun": lambda x, A=A, b=b: A @ x - b})
    try:
        res = sopt.minimize(
            fun,
            x0,
            method="SLSQP",
            jac=jac,
            constraints=cons,
            options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
        )
        x = np.asarray(res.x, dtype=float)
        if res.success and _feasible(x, G, h, A, b):
            return _active_set_polish(x, P, q, G, h, A, b)
    except Exception:
        pass
    return None

def _solve_cvxpy(
    P: np.ndarray,
    q: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    if not _HAS_CVXPY:
        raise ValueError("No available QP backend")
    x = cp.Variable(P.shape[0])
    cons = []
    if G.shape[0]:
        cons.append(G @ x <= h)
    if A.shape[0]:
        cons.append(A @ x == b)
    obj = 0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, verbose=False)
    if x.value is None:
        raise ValueError(f"Solver failed (status={prob.status})")
    return np.asarray(x.value, dtype=float).reshape(-1)

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        P = _get_P(problem)
        q = _as_vec(problem["q"])
        n = P.shape[0]
        P = np.asarray(P, dtype=float).reshape(n, n)
        P = 0.5 * (P + P.T)

        G = _as_mat(problem.get("G", []), n)
        h = _as_vec(problem.get("h", []))
        A = _as_mat(problem.get("A", []), n)
        b = _as_vec(problem.get("b", []))

        if G.shape[0] != h.shape[0]:
            h = h.reshape(G.shape[0])
        if A.shape[0] != b.shape[0]:
            b = b.reshape(A.shape[0])

        x = None

        if G.shape[0] == 0:
            x = _solve_kkt(P, q, A, b)
            if not _feasible(x, G, h, A, b):
                x = None
        else:
            x = _solve_active_set_small(P, q, G, h, A, b)
            if x is None:
                x = _solve_osqp(P, q, G, h, A, b)

        if x is None:
            x = _solve_scipy(P, q, G, h, A, b)
        if x is None:
            x = _solve_cvxpy(P, q, G, h, A, b)

        return {"solution": x.tolist(), "objective": _objective(P, q, x)}