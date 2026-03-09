from typing import Any

import numba as nb
import numpy as np
from scipy.optimize import linprog

@nb.njit(cache=True)
def _solve_linear_system(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    n = b.shape[0]
    M = A.copy()
    x = b.copy()

    for k in range(n):
        piv = k
        best = abs(M[k, k])
        for i in range(k + 1, n):
            v = abs(M[i, k])
            if v > best:
                best = v
                piv = i
        if best < 1e-14:
            return np.empty(0, dtype=np.float64), 0
        if piv != k:
            for j in range(k, n):
                tmp = M[k, j]
                M[k, j] = M[piv, j]
                M[piv, j] = tmp
            tmpb = x[k]
            x[k] = x[piv]
            x[piv] = tmpb

        pivot = M[k, k]
        for i in range(k + 1, n):
            factor = M[i, k] / pivot
            M[i, k] = 0.0
            for j in range(k + 1, n):
                M[i, j] -= factor * M[k, j]
            x[i] -= factor * x[k]

    sol = np.empty(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = x[i]
        for j in range(i + 1, n):
            s -= M[i, j] * sol[j]
        denom = M[i, i]
        if abs(denom) < 1e-14:
            return np.empty(0, dtype=np.float64), 0
        sol[i] = s / denom
    return sol, 1

@nb.njit(cache=True)
def _core_solve(
    G: np.ndarray,
    sigma: np.ndarray,
    p_min: np.ndarray,
    p_max: np.ndarray,
    s_min: float,
) -> tuple[np.ndarray, int]:
    n = G.shape[0]
    tol = 1e-10

    if n == 0:
        return np.empty(0, dtype=np.float64), 1

    diag = np.empty(n, dtype=np.float64)
    for i in range(n):
        d = G[i, i]
        if d <= 0.0:
            return np.empty(0, dtype=np.float64), -1
        diag[i] = d

    if n == 1:
        p = s_min * sigma[0] / diag[0]
        if p < p_min[0]:
            p = p_min[0]
        if p > p_max[0] + 1e-9:
            return np.empty(0, dtype=np.float64), 0
        out = np.empty(1, dtype=np.float64)
        out[0] = p
        return out, 1

    F = np.empty((n, n), dtype=np.float64)
    u = np.empty(n, dtype=np.float64)
    for i in range(n):
        inv = s_min / diag[i]
        u[i] = sigma[i] * inv
        for j in range(n):
            if i == j:
                F[i, j] = 0.0
            else:
                F[i, j] = G[i, j] * inv

    # Cheapest path: p_min already feasible.
    req_min = np.empty(n, dtype=np.float64)
    feasible_min = True
    for i in range(n):
        s = u[i]
        for j in range(n):
            s += F[i, j] * p_min[j]
        req_min[i] = s
        if p_min[i] + 1e-8 < s or p_min[i] > p_max[i] + 1e-8:
            feasible_min = False
    if feasible_min:
        return p_min.copy(), 1

    # One fixed-point step often solves the instance.
    p1 = np.empty(n, dtype=np.float64)
    ok = True
    for i in range(n):
        v = req_min[i]
        if v < p_min[i]:
            v = p_min[i]
        p1[i] = v
        if v > p_max[i] + 1e-8:
            ok = False
    if ok:
        fixed = True
        for i in range(n):
            s = u[i]
            for j in range(n):
                s += F[i, j] * p1[j]
            if p1[i] + 1e-8 < s:
                fixed = False
                break
        if fixed:
            return p1, 1

    # Unconstrained least solution.
    A = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            A[i, j] = -F[i, j]
        A[i, i] += 1.0

    p0, ok0 = _solve_linear_system(A, u)
    if ok0 == 0:
        return np.empty(0, dtype=np.float64), 0

    for i in range(n):
        if p0[i] > p_max[i] + 1e-8:
            return np.empty(0, dtype=np.float64), 0

    active = np.zeros(n, dtype=np.uint8)
    any_below = False
    for i in range(n):
        if p0[i] < p_min[i] - tol:
            active[i] = 1
            any_below = True

    if not any_below:
        return p0, 1

    p = p_min.copy()

    for _ in range(n):
        free_count = 0
        for i in range(n):
            if active[i] == 0:
                free_count += 1

        if free_count == 0:
            break

        ff = np.empty(free_count, dtype=np.int64)
        aa = np.empty(n - free_count, dtype=np.int64)
        fi = 0
        ai = 0
        for i in range(n):
            if active[i] == 0:
                ff[fi] = i
                fi += 1
            else:
                aa[ai] = i
                ai += 1

        M = np.empty((free_count, free_count), dtype=np.float64)
        rhs = np.empty(free_count, dtype=np.float64)
        for r in range(free_count):
            ir = ff[r]
            s = u[ir]
            for k in range(aa.shape[0]):
                s += F[ir, aa[k]] * p_min[aa[k]]
            rhs[r] = s
            for c in range(free_count):
                ic = ff[c]
                v = -F[ir, ic]
                if r == c:
                    v += 1.0
                M[r, c] = v

        p_free, okf = _solve_linear_system(M, rhs)
        if okf == 0:
            return np.empty(0, dtype=np.float64), 0

        for i in range(n):
            if active[i] != 0:
                p[i] = p_min[i]
        for r in range(free_count):
            p[ff[r]] = p_free[r]

        changed = False
        for i in range(n):
            if active[i] == 0 and p[i] < p_min[i] - tol:
                active[i] = 1
                changed = True
                p[i] = p_min[i]

        if not changed:
            for i in range(n):
                if p[i] > p_max[i] + 1e-8:
                    return np.empty(0, dtype=np.float64), 0
            for i in range(n):
                s = u[i]
                for j in range(n):
                    s += F[i, j] * p[j]
                if p[i] + 1e-8 < s:
                    return np.empty(0, dtype=np.float64), 0
            return p, 1

    # Short monotone refinement as a last cheap attempt.
    p = p1.copy()
    for _ in range(64):
        diff = 0.0
        for i in range(n):
            s = u[i]
            for j in range(n):
                s += F[i, j] * p[j]
            v = s
            if v < p_min[i]:
                v = p_min[i]
            d = abs(v - p[i])
            if d > diff:
                diff = d
            p[i] = v
            if v > p_max[i] + 1e-8:
                return np.empty(0, dtype=np.float64), 0
        if diff <= 1e-11:
            return p, 1

    return np.empty(0, dtype=np.float64), 0

class Solver:
    def __init__(self) -> None:
        dummy_G = np.eye(1, dtype=np.float64)
        dummy_v = np.ones(1, dtype=np.float64)
        _solve_linear_system(dummy_G, dummy_v)
        _core_solve(dummy_G, dummy_v, dummy_v, dummy_v * 2.0, 0.1)

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        G_raw = problem["G"]
        sigma_raw = problem["σ"]
        pmin_raw = problem["P_min"]
        pmax_raw = problem["P_max"]
        s_min = float(problem["S_min"])

        n = len(pmin_raw)
        if n == 0:
            return {"P": []}

        if n == 1:
            d = float(G_raw[0][0])
            if d <= 0.0:
                raise ValueError("Solver failed (invalid diagonal gains)")
            p = s_min * float(sigma_raw[0]) / d
            pmin0 = float(pmin_raw[0])
            if p < pmin0:
                p = pmin0
            if p > float(pmax_raw[0]) + 1e-9:
                raise ValueError("Solver failed (infeasible)")
            return {"P": [p]}

        if n == 2:
            d1 = float(G_raw[0][0])
            d2 = float(G_raw[1][1])
            if d1 <= 0.0 or d2 <= 0.0:
                raise ValueError("Solver failed (invalid diagonal gains)")

            p1min = float(pmin_raw[0])
            p2min = float(pmin_raw[1])
            p1max = float(pmax_raw[0])
            p2max = float(pmax_raw[1])

            a12 = s_min * float(G_raw[0][1]) / d1
            a21 = s_min * float(G_raw[1][0]) / d2
            u1 = s_min * float(sigma_raw[0]) / d1
            u2 = s_min * float(sigma_raw[1]) / d2
            tol = 1e-8

            req1 = u1 + a12 * p2min
            req2 = u2 + a21 * p1min
            if p1min + tol >= req1 and p2min + tol >= req2:
                if p1min <= p1max + tol and p2min <= p2max + tol:
                    return {"P": [p1min, p2min]}
                raise ValueError("Solver failed (infeasible)")

            best_p1 = 0.0
            best_p2 = 0.0
            best_obj = 0.0
            found = False

            for mask in range(4):
                act1 = mask & 1
                act2 = mask & 2

                if act1:
                    p1 = p1min
                    if act2:
                        p2 = p2min
                    else:
                        p2 = u2 + a21 * p1
                elif act2:
                    p2 = p2min
                    p1 = u1 + a12 * p2
                else:
                    den = 1.0 - a12 * a21
                    if abs(den) < 1e-14:
                        continue
                    p1 = (u1 + a12 * u2) / den
                    p2 = (u2 + a21 * u1) / den

                if (
                    p1 < p1min - tol
                    or p2 < p2min - tol
                    or p1 > p1max + tol
                    or p2 > p2max + tol
                ):
                    continue

                r1 = u1 + a12 * p2
                r2 = u2 + a21 * p1

                if act1:
                    if p1min + tol < r1:
                        continue
                elif abs(p1 - r1) > 1e-7:
                    continue

                if act2:
                    if p2min + tol < r2:
                        continue
                elif abs(p2 - r2) > 1e-7:
                    continue

                obj = p1 + p2
                if (not found) or (obj < best_obj):
                    found = True
                    best_obj = obj
                    best_p1 = p1
                    best_p2 = p2

            if found:
                return {"P": [best_p1, best_p2]}

        G = np.asarray(G_raw, dtype=np.float64)
        sigma = np.asarray(sigma_raw, dtype=np.float64)
        p_min = np.asarray(pmin_raw, dtype=np.float64)
        p_max = np.asarray(pmax_raw, dtype=np.float64)

        p, status = _core_solve(G, sigma, p_min, p_max, s_min)
        if status == 1:
            return {"P": p}

        n = G.shape[0]
        if n == 0:
            return {"P": []}

        diag = np.diag(G).astype(float, copy=False)
        if np.any(diag <= 0):
            raise ValueError("Solver failed (invalid diagonal gains)")

        idx = np.arange(n)
        A_ub = s_min * G.copy()
        A_ub[idx, idx] = -diag
        b_ub = -s_min * sigma
        bounds = list(zip(p_min.tolist(), p_max.tolist()))

        res = linprog(
            np.ones(n, dtype=float),
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            raise ValueError(f"Solver failed ({res.message})")

        return {"P": np.asarray(res.x, dtype=float)}