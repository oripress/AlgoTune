from __future__ import annotations

import math
from contextlib import nullcontext

import numpy as np
from scipy.linalg import eigvals as scipy_eigvals

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional dependency
    threadpool_limits = None
    threadpool_limits = None

def _single_thread_blas():
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=1)

def _trim_leading(coeffs: list[float]) -> list[float]:
    i = 0
    n = len(coeffs)
    while i < n - 1 and coeffs[i] == 0:
        i += 1
    return [float(x) for x in coeffs[i:]]

def _horner_and_derivative(coeffs: list[float], x: float) -> tuple[float, float]:
    p = coeffs[0]
    dp = 0.0
    for a in coeffs[1:]:
        dp = dp * x + p
        p = p * x + a
    return p, dp

def _polyval(coeffs: list[float], x: float) -> float:
    p = coeffs[0]
    for a in coeffs[1:]:
        p = p * x + a
    return p

def _quadratic_roots(coeffs: list[float]) -> list[float]:
    a, b, c = coeffs
    if a == 0:
        return [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0.0 and disc > -1e-15 * (b * b + abs(a * c) + 1.0):
        disc = 0.0
    disc = max(disc, 0.0)
    s = math.sqrt(disc)
    if s == 0.0:
        r = -0.5 * b / a
        return [r, r]
    q = -0.5 * (b + math.copysign(s, b))
    r1 = q / a
    r2 = c / q if q != 0.0 else (-b - s) / (2.0 * a)
    if r1 <= r2:
        return [r1, r2]
    return [r2, r1]

def _cubic_roots(coeffs: list[float]) -> list[float]:
    a, b, c, d = coeffs
    if a == 0.0:
        return _quadratic_roots([b, c, d])

    inv_a = 1.0 / a
    aa = b * inv_a
    bb = c * inv_a
    cc = d * inv_a

    shift = aa / 3.0
    p = bb - aa * aa / 3.0
    q = (2.0 * aa * aa * aa) / 27.0 - (aa * bb) / 3.0 + cc

    if abs(p) < 1e-18 and abs(q) < 1e-18:
        r = -shift
        return [r, r, r]

    third = 1.0 / 3.0
    half_q = 0.5 * q
    disc = half_q * half_q + (p * third) * (p * third) * (p * third)

    if disc > 1e-14:
        roots = np.roots(np.asarray(coeffs, dtype=float)).real
        roots.sort()
        return roots.tolist()

    if disc < 0.0:
        disc = 0.0

    if abs(p) < 1e-18:
        r = math.copysign(abs(-q) ** third, -q) - shift
        return [r, r, r]

    rad = math.sqrt(max(0.0, -p / 3.0))
    denom = rad * rad * rad
    if denom == 0.0:
        r = -shift
        return [r, r, r]

    arg = -q / (2.0 * denom)
    arg = max(-1.0, min(1.0, arg))
    phi = math.acos(arg)
    t = 2.0 * rad
    roots = [
        t * math.cos(phi / 3.0) - shift,
        t * math.cos((phi + 2.0 * math.pi) / 3.0) - shift,
        t * math.cos((phi + 4.0 * math.pi) / 3.0) - shift,
    ]
    roots.sort()
    return roots

def _group_sorted(values: list[float], tol: float = 1e-9) -> list[tuple[float, int]]:
    if not values:
        return []
    groups: list[tuple[float, int]] = []
    cur = values[0]
    cnt = 1
    for v in values[1:]:
        scale = max(1.0, abs(cur), abs(v))
        if abs(v - cur) <= tol * scale:
            cur = (cur * cnt + v) / (cnt + 1)
            cnt += 1
        else:
            groups.append((cur, cnt))
            cur = v
            cnt = 1
    groups.append((cur, cnt))
    return groups

def _sign_with_tol(value: float, scale: float) -> int:
    tol = 1e-12 * max(1.0, scale)
    if value > tol:
        return 1
    if value < -tol:
        return -1
    return 0

def _bisect_root(coeffs: list[float], left: float, right: float, f_left: float, f_right: float) -> float:
    if f_left == 0.0:
        return left
    if f_right == 0.0:
        return right

    x = 0.5 * (left + right)
    fx = _polyval(coeffs, x)
    for _ in range(64):
        if fx == 0.0:
            return x
        if f_left * fx <= 0.0:
            right = x
            f_right = fx
        else:
            left = x
            f_left = fx
        nx = 0.5 * (left + right)
        if nx == x or abs(right - left) <= 1e-15 * max(1.0, abs(nx)):
            return nx
        x = nx
        fx = _polyval(coeffs, x)
    return x

def _outer_bracket_left(coeffs: list[float], start: float, f_start: float, sign_inf: int) -> tuple[float, float, float]:
    step = 1.0
    x = start - step
    fx = _polyval(coeffs, x)
    sx = _sign_with_tol(fx, abs(fx))
    for _ in range(80):
        if sx == 0 or sx == sign_inf or fx * f_start <= 0.0:
            return x, start, fx
        step *= 2.0
        x = start - step
        fx = _polyval(coeffs, x)
        sx = _sign_with_tol(fx, abs(fx))
    return x, start, fx

def _outer_bracket_right(coeffs: list[float], start: float, f_start: float, sign_inf: int) -> tuple[float, float, float]:
    step = 1.0
    x = start + step
    fx = _polyval(coeffs, x)
    sx = _sign_with_tol(fx, abs(fx))
    for _ in range(80):
        if sx == 0 or sx == sign_inf or fx * f_start <= 0.0:
            return start, x, fx
        step *= 2.0
        x = start + step
        fx = _polyval(coeffs, x)
        sx = _sign_with_tol(fx, abs(fx))
    return start, x, fx

def _recursive_all_real_roots(coeffs: list[float]) -> list[float]:
    degree = len(coeffs) - 1
    if degree <= 0:
        return []
    if degree == 1:
        return [-coeffs[1] / coeffs[0]]
    if degree == 2:
        return _quadratic_roots(coeffs)
    if degree == 3:
        return _cubic_roots(coeffs)

    dcoeffs = [coeffs[i] * (degree - i) for i in range(degree)]
    crit = _recursive_all_real_roots(dcoeffs)
    crit.sort()
    crit_groups = _group_sorted(crit)

    roots: list[float] = []

    distinct = [c for c, _ in crit_groups]
    f_crit = [_polyval(coeffs, c) for c in distinct]
    coeff_scale = sum(abs(x) for x in coeffs)

    for (c, mult), fc in zip(crit_groups, f_crit):
        s = _sign_with_tol(fc, coeff_scale)
        if s == 0:
            roots.extend([c] * (mult + 1))

    if not distinct:
        sign_left_inf = 1 if coeffs[0] > 0.0 and degree % 2 == 0 else -1
        sign_right_inf = 1 if coeffs[0] > 0.0 else -1
        x0 = 0.0
        f0 = _polyval(coeffs, x0)
        s0 = _sign_with_tol(f0, coeff_scale)
        if s0 == 0:
            roots.append(x0)
            return roots
        if s0 != sign_left_inf:
            left, right, f_left = _outer_bracket_left(coeffs, x0, f0, sign_left_inf)
            roots.append(_bisect_root(coeffs, left, right, f_left, f0))
        else:
            left, right, f_right = _outer_bracket_right(coeffs, x0, f0, sign_right_inf)
            roots.append(_bisect_root(coeffs, left, right, f0, f_right))
        roots.sort()
        return roots

    sign_left_inf = 1 if coeffs[0] > 0.0 and degree % 2 == 0 else -1
    sign_right_inf = 1 if coeffs[0] > 0.0 else -1

    s_first = _sign_with_tol(f_crit[0], coeff_scale)
    if s_first != 0 and s_first != sign_left_inf:
        left, right, f_left = _outer_bracket_left(coeffs, distinct[0], f_crit[0], sign_left_inf)
        roots.append(_bisect_root(coeffs, left, right, f_left, f_crit[0]))

    for i in range(len(distinct) - 1):
        fa = f_crit[i]
        fb = f_crit[i + 1]
        sa = _sign_with_tol(fa, coeff_scale)
        sb = _sign_with_tol(fb, coeff_scale)
        if sa != 0 and sb != 0 and sa != sb:
            roots.append(_bisect_root(coeffs, distinct[i], distinct[i + 1], fa, fb))

    s_last = _sign_with_tol(f_crit[-1], coeff_scale)
    if s_last != 0 and s_last != sign_right_inf:
        left, right, f_right = _outer_bracket_right(coeffs, distinct[-1], f_crit[-1], sign_right_inf)
        roots.append(_bisect_root(coeffs, left, right, f_crit[-1], f_right))

    roots.sort()
    return roots

def _polish_grouped_roots(coeffs: list[float], roots: list[float]) -> list[float]:
    groups = _group_sorted(sorted(roots), tol=1e-8)
    polished: list[float] = []
    for x, mult in groups:
        cur = x
        for _ in range(10):
            fx, dfx = _horner_and_derivative(coeffs, cur)
            if dfx == 0.0:
                break
            step = mult * fx / dfx
            nxt = cur - step
            if not math.isfinite(nxt):
                break
            cur = nxt
            if abs(step) <= 1e-15 * max(1.0, abs(cur)):
                break
        polished.extend([cur] * mult)
    polished.sort()
    return polished

class Solver:
    def solve(self, problem, **kwargs):
        coeffs = problem
        n = len(coeffs)
        i = 0
        while i < n - 1 and coeffs[i] == 0:
            i += 1
        if i:
            coeffs = [float(x) for x in coeffs[i:]]
        degree = len(coeffs) - 1
        if degree <= 0:
            return []

        if degree == 1:
            a, b = coeffs
            return [-b / a]

        if degree == 2:
            roots = _quadratic_roots(coeffs)
            roots.sort(reverse=True)
            return roots

        if degree == 3:
            roots = _cubic_roots(coeffs)
            roots.sort(reverse=True)
            return roots

        roots = np.roots(coeffs).real
        roots.sort()
        return roots[::-1].tolist()