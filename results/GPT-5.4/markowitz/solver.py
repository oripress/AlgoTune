from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit
from scipy.optimize import minimize

@njit(cache=True, fastmath=True)
def _kkt_ok_nb(mu: np.ndarray, sigma: np.ndarray, gamma: float, w: np.ndarray) -> bool:
    n = w.shape[0]
    s = 0.0
    for i in range(n):
        wi = w[i]
        if not np.isfinite(wi) or wi < -1e-7:
            return False
        s += wi
    if abs(s - 1.0) > 2e-6:
        return False

    two_g = 2.0 * gamma
    q = np.empty(n, dtype=np.float64)
    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc += sigma[i, j] * w[j]
        q[i] = mu[i] - two_g * acc

    lam_min = 1e300
    lam_max = -1e300
    has_active = False
    for i in range(n):
        if w[i] > 1e-9:
            has_active = True
            qi = q[i]
            if qi < lam_min:
                lam_min = qi
            if qi > lam_max:
                lam_max = qi

    if has_active:
        if lam_max - lam_min > 2e-7:
            return False
        lam = 0.5 * (lam_min + lam_max)
    else:
        lam = -1e300
        for i in range(n):
            if q[i] > lam:
                lam = q[i]

    thresh = lam + 2e-7
    for i in range(n):
        if w[i] <= 1e-9 and q[i] > thresh:
            return False
    return True

@njit(cache=True, fastmath=True)
def _active_set_solve_nb(mu: np.ndarray, sigma: np.ndarray, gamma: float) -> np.ndarray:
    n = mu.shape[0]
    if n == 1:
        out = np.empty(1, dtype=np.float64)
        out[0] = 1.0
        return out

    active = np.ones(n, dtype=np.uint8)

    best_vertex = 0
    best_score = mu[0] - gamma * sigma[0, 0]
    for i in range(1, n):
        score = mu[i] - gamma * sigma[i, i]
        if score > best_score:
            best_score = score
            best_vertex = i

    two_g = 2.0 * gamma

    for _ in range(4 * n + 20):
        k = 0
        for i in range(n):
            if active[i] != 0:
                k += 1

        if k == 0:
            active[best_vertex] = 1
            k = 1

        idx = np.empty(k, dtype=np.int64)
        p = 0
        for i in range(n):
            if active[i] != 0:
                idx[p] = i
                p += 1

        if k == 1:
            w = np.zeros(n, dtype=np.float64)
            w[idx[0]] = 1.0
            return w

        kkt = np.zeros((k + 1, k + 1), dtype=np.float64)
        rhs = np.empty(k + 1, dtype=np.float64)

        scale = 0.0
        for i in range(k):
            ii = idx[i]
            rhs[i] = mu[ii]
            for j in range(k):
                val = two_g * sigma[ii, idx[j]]
                kkt[i, j] = val
            d = abs(kkt[i, i])
            if d > scale:
                scale = d
            kkt[i, k] = 1.0
            kkt[k, i] = 1.0
        rhs[k] = 1.0

        eps = 1e-12 * (1.0 + scale)
        for i in range(k):
            kkt[i, i] += eps

        z = np.linalg.solve(kkt, rhs)

        has_negative = False
        for i in range(k):
            if z[i] < -1e-11:
                active[idx[i]] = 0
                has_negative = True
        if has_negative:
            continue

        w = np.zeros(n, dtype=np.float64)
        total = 0.0
        for i in range(k):
            wi = z[i]
            if wi < 0.0:
                wi = 0.0
            ii = idx[i]
            w[ii] = wi
            total += wi

        if total <= 0.0:
            return np.empty(0, dtype=np.float64)

        if abs(total - 1.0) > 1e-12:
            inv_total = 1.0 / total
            for i in range(n):
                w[i] *= inv_total

        q = np.empty(n, dtype=np.float64)
        for i in range(n):
            acc = 0.0
            for j in range(n):
                acc += sigma[i, j] * w[j]
            q[i] = mu[i] - two_g * acc

        lam = 0.0
        cnt = 0
        for i in range(k):
            ii = idx[i]
            if w[ii] > 1e-9:
                lam += q[ii]
                cnt += 1
        if cnt == 0:
            lam = q[idx[0]]
        else:
            lam /= cnt

        vmax = -1e300
        add = -1
        for i in range(n):
            if active[i] == 0:
                v = q[i] - lam
                if v > vmax:
                    vmax = v
                    add = i

        if add == -1 or vmax <= 2e-8:
            return w

        active[add] = 1

    return np.empty(0, dtype=np.float64)

@njit(cache=True)
def _solve_fast_nb(mu: np.ndarray, sigma: np.ndarray, gamma: float) -> np.ndarray:
    w = _active_set_solve_nb(mu, sigma, gamma)
    if w.shape[0] == mu.shape[0] and _kkt_ok_nb(mu, sigma, gamma, w):
        return w
    return np.empty(0, dtype=np.float64)

def _objective(mu: np.ndarray, sigma: np.ndarray, gamma: float, w: np.ndarray) -> float:
    return float(mu @ w - gamma * (w @ sigma @ w))

def _project_simplex(v: np.ndarray) -> np.ndarray:
    n = v.size
    if n == 1:
        return np.array([1.0], dtype=float)

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0.0
    if not np.any(cond):
        return np.full(n, 1.0 / n, dtype=float)

    rho = int(ind[cond][-1])
    theta = cssv[rho - 1] / rho
    w = v - theta
    np.maximum(w, 0.0, out=w)
    s = float(w.sum())
    if s > 0.0:
        w /= s
    else:
        w.fill(1.0 / n)
    return w

def _kkt_ok_py(mu: np.ndarray, sigma: np.ndarray, gamma: float, w: np.ndarray) -> bool:
    if w.ndim != 1 or not np.isfinite(w).all():
        return False
    if np.any(w < -1e-7):
        return False
    if abs(float(w.sum()) - 1.0) > 2e-6:
        return False

    q = mu - 2.0 * gamma * (sigma @ w)
    active = w > 1e-9
    if np.any(active):
        lam_min = float(np.min(q[active]))
        lam_max = float(np.max(q[active]))
        if lam_max - lam_min > 2e-7:
            return False
        lam = 0.5 * (lam_min + lam_max)
    else:
        lam = float(np.max(q))

    return bool(np.all(q[~active] <= lam + 2e-7))

def _solve_on_support_py(
    mu: np.ndarray, sigma: np.ndarray, gamma: float, idx: np.ndarray
) -> tuple[np.ndarray, float] | None:
    k = idx.size
    if k == 0:
        return None
    if k == 1:
        i = int(idx[0])
        return np.array([1.0], dtype=float), float(mu[i] - 2.0 * gamma * sigma[i, i])

    sig = sigma[np.ix_(idx, idx)]
    mu_s = mu[idx]
    a = 2.0 * gamma * sig
    ones = np.ones(k, dtype=float)

    try:
        rhs = np.empty((k, 2), dtype=float)
        rhs[:, 0] = ones
        rhs[:, 1] = mu_s
        sol = np.linalg.solve(a, rhs)
        x = sol[:, 0]
        y = sol[:, 1]
        denom = float(ones @ x)
        if abs(denom) > 1e-15:
            lam = float(((ones @ y) - 1.0) / denom)
            w_s = y - lam * x
            return w_s, lam
    except np.linalg.LinAlgError:
        pass

    kkt = np.empty((k + 1, k + 1), dtype=float)
    kkt[:k, :k] = a
    kkt[:k, k] = 1.0
    kkt[k, :k] = 1.0
    kkt[k, k] = 0.0

    rhs2 = np.empty(k + 1, dtype=float)
    rhs2[:k] = mu_s
    rhs2[k] = 1.0

    try:
        z = np.linalg.solve(kkt, rhs2)
    except np.linalg.LinAlgError:
        z, *_ = np.linalg.lstsq(kkt, rhs2, rcond=1e-12)

    return z[:k], float(z[k])

def _active_set_solve_py(mu: np.ndarray, sigma: np.ndarray, gamma: float) -> np.ndarray | None:
    n = mu.size
    if n == 1:
        return np.array([1.0], dtype=float)

    active = np.ones(n, dtype=bool)
    diag = np.diag(sigma)
    best_vertex = int(np.argmax(mu - gamma * diag))

    for _ in range(4 * n + 20):
        idx = np.flatnonzero(active)
        solved = _solve_on_support_py(mu, sigma, gamma, idx)
        if solved is None:
            return None
        w_s, lam = solved

        negative = w_s < -1e-11
        if np.any(negative):
            active[idx[negative]] = False
            if not np.any(active):
                active[best_vertex] = True
            continue

        w = np.zeros(n, dtype=float)
        w[idx] = w_s
        w[w < 0.0] = 0.0
        total = float(w.sum())
        if total <= 0.0:
            return None
        if abs(total - 1.0) > 1e-12:
            w /= total

        q = mu - 2.0 * gamma * (sigma @ w)
        inactive = ~active
        if not np.any(inactive):
            return w

        viol = q[inactive] - lam
        vmax = float(np.max(viol))
        if vmax <= 2e-8:
            return w

        add_idx = np.flatnonzero(inactive)[int(np.argmax(viol))]
        active[add_idx] = True

    return None

def _pgd_solve(mu: np.ndarray, sigma: np.ndarray, gamma: float) -> np.ndarray:
    n = mu.size
    if n == 1:
        return np.array([1.0], dtype=float)

    w = np.zeros(n, dtype=float)
    w[int(np.argmax(mu - gamma * np.diag(sigma)))] = 1.0

    lipschitz_bound = float(np.max(np.sum(np.abs(sigma), axis=1)))
    step = 1.0 / max(2.0 * gamma * max(lipschitz_bound, 0.0), 1e-12)
    best_obj = _objective(mu, sigma, gamma, w)

    for _ in range(1500):
        grad = mu - 2.0 * gamma * (sigma @ w)
        cand = _project_simplex(w + step * grad)
        cand_obj = _objective(mu, sigma, gamma, cand)

        if cand_obj + 1e-15 < best_obj:
            step *= 0.5
            if step < 1e-15:
                break
            continue

        diff = float(np.max(np.abs(cand - w)))
        w = cand
        best_obj = cand_obj
        if diff < 1e-12:
            break

    return w

def _slsqp_solve(
    mu: np.ndarray, sigma: np.ndarray, gamma: float, x0: np.ndarray | None = None
) -> np.ndarray | None:
    n = mu.size
    if x0 is None:
        x0 = np.full(n, 1.0 / n, dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float).copy()
        x0[x0 < 0.0] = 0.0
        s0 = float(x0.sum())
        if s0 <= 0.0:
            x0.fill(1.0 / n)
        else:
            x0 /= s0

    ones = np.ones(n, dtype=float)

    def fun(w: np.ndarray) -> float:
        return float(gamma * (w @ sigma @ w) - mu @ w)

    def jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * gamma * (sigma @ w) - mu

    res = minimize(
        fun,
        x0,
        method="SLSQP",
        jac=jac,
        bounds=[(0.0, None)] * n,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0), "jac": lambda w: ones}],
        options={"ftol": 1e-12, "maxiter": 200, "disp": False},
    )
    if res.x is None or not np.isfinite(res.x).all():
        return None

    w = np.asarray(res.x, dtype=float)
    w[w < 0.0] = 0.0
    s = float(w.sum())
    if s <= 0.0:
        return None
    w /= s
    return w

class Solver:
    def __init__(self) -> None:
        try:
            mu = np.array([0.1, 0.2], dtype=np.float64)
            sigma = np.array([[1.0, 0.1], [0.1, 1.2]], dtype=np.float64)
            _ = _solve_fast_nb(mu, sigma, 1.0)
        except Exception:
            pass

    def solve(self, problem, **kwargs) -> Any:
        mu = np.asarray(problem["μ"], dtype=np.float64)
        sigma = np.asarray(problem["Σ"], dtype=np.float64)
        gamma = float(problem["γ"])

        if mu.ndim != 1:
            return None
        n = mu.size
        if n == 0:
            return None
        if sigma.ndim != 2 or sigma.shape[0] != n or sigma.shape[1] != n:
            return None
        if gamma <= 0.0:
            return None

        w = _solve_fast_nb(mu, sigma, gamma)
        if w.size == n:
            return {"w": w}

        w = _active_set_solve_py(mu, sigma, gamma)
        if w is not None and _kkt_ok_py(mu, sigma, gamma, w):
            return {"w": w}

        w = _pgd_solve(mu, sigma, gamma)
        if not _kkt_ok_py(mu, sigma, gamma, w):
            w = _slsqp_solve(mu, sigma, gamma, x0=w)
            if w is None:
                return None

        w[w < 0.0] = 0.0
        s = float(w.sum())
        if s <= 0.0:
            return None
        if abs(s - 1.0) > 1e-12:
            w /= s
        return {"w": w}