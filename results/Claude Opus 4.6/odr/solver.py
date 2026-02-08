import numpy as np
from numba import njit
import math


@njit(cache=True, fastmath=True)
def _solve_odr_numba(x, y, sx, sy):
    n = len(x)
    sx2 = np.empty(n)
    sy2 = np.empty(n)
    sum_x = 0.0
    sum_y = 0.0
    for i in range(n):
        sx2[i] = sx[i] * sx[i]
        sy2[i] = sy[i] * sy[i]
        sum_x += x[i]
        sum_y += y[i]
    
    xm = sum_x / n
    ym = sum_y / n
    num = 0.0
    den = 0.0
    for i in range(n):
        dx = x[i] - xm
        num += dx * (y[i] - ym)
        den += dx * dx
    if den > 0.0:
        a_ols = num / den
    else:
        a_ols = 1.0
    
    a_prev = a_ols * 0.9 - 0.1
    a_curr = a_ols
    
    a2 = a_prev * a_prev
    sw = 0.0; swx = 0.0; swy = 0.0
    for i in range(n):
        w = 1.0 / (sy2[i] + a2 * sx2[i])
        sw += w; swx += w * x[i]; swy += w * y[i]
    b = (swy - a_prev * swx) / sw
    g_prev = 0.0
    for i in range(n):
        w = 1.0 / (sy2[i] + a2 * sx2[i])
        r = y[i] - a_prev * x[i] - b
        g_prev += x[i] * w * r + a_prev * sx2[i] * w * w * r * r
    
    a2 = a_curr * a_curr
    sw = 0.0; swx = 0.0; swy = 0.0
    for i in range(n):
        w = 1.0 / (sy2[i] + a2 * sx2[i])
        sw += w; swx += w * x[i]; swy += w * y[i]
    b = (swy - a_curr * swx) / sw
    g_curr = 0.0
    for i in range(n):
        w = 1.0 / (sy2[i] + a2 * sx2[i])
        r = y[i] - a_curr * x[i] - b
        g_curr += x[i] * w * r + a_curr * sx2[i] * w * w * r * r
    
    for _ in range(50):
        dg = g_curr - g_prev
        if abs(dg) < 1e-30:
            break
        a_new = a_curr - g_curr * (a_curr - a_prev) / dg
        a_prev = a_curr
        g_prev = g_curr
        a_curr = a_new
        
        a2 = a_curr * a_curr
        sw = 0.0; swx = 0.0; swy = 0.0
        for i in range(n):
            w = 1.0 / (sy2[i] + a2 * sx2[i])
            sw += w; swx += w * x[i]; swy += w * y[i]
        b = (swy - a_curr * swx) / sw
        g_curr = 0.0
        for i in range(n):
            w = 1.0 / (sy2[i] + a2 * sx2[i])
            r = y[i] - a_curr * x[i] - b
            g_curr += x[i] * w * r + a_curr * sx2[i] * w * w * r * r
        
        if abs(a_curr - a_prev) < 1e-13 * (1.0 + abs(a_curr)):
            break
    
    a2 = a_curr * a_curr
    sw = 0.0; swx = 0.0; swy = 0.0
    for i in range(n):
        w = 1.0 / (sy2[i] + a2 * sx2[i])
        sw += w; swx += w * x[i]; swy += w * y[i]
    b_opt = (swy - a_curr * swx) / sw
    
    return a_curr, b_opt


def _solve_odr_python(xl, yl, sxl, syl):
    """Pure Python solver for small n, avoids numpy overhead."""
    n = len(xl)
    sx2 = [s * s for s in sxl]
    sy2 = [s * s for s in syl]
    
    sum_x = sum(xl)
    sum_y = sum(yl)
    xm = sum_x / n
    ym = sum_y / n
    num = 0.0
    den = 0.0
    for i in range(n):
        dx = xl[i] - xm
        num += dx * (yl[i] - ym)
        den += dx * dx
    a_ols = num / den if den > 0 else 1.0
    
    a_prev = a_ols * 0.9 - 0.1
    a_curr = a_ols
    
    # Gradient at a_prev
    a2 = a_prev * a_prev
    sw = swx = swy = 0.0
    for i in range(n):
        w = 1.0 / (sy2[i] + a2 * sx2[i])
        sw += w; swx += w * xl[i]; swy += w * yl[i]
    b = (swy - a_prev * swx) / sw
    g_prev = 0.0
    for i in range(n):
        w = 1.0 / (sy2[i] + a2 * sx2[i])
        r = yl[i] - a_prev * xl[i] - b
        g_prev += xl[i] * w * r + a_prev * sx2[i] * w * w * r * r
    
    # Gradient at a_curr
    a2 = a_curr * a_curr
    sw = swx = swy = 0.0
    for i in range(n):
        w = 1.0 / (sy2[i] + a2 * sx2[i])
        sw += w; swx += w * xl[i]; swy += w * yl[i]
    b = (swy - a_curr * swx) / sw
    g_curr = 0.0
    for i in range(n):
        w = 1.0 / (sy2[i] + a2 * sx2[i])
        r = yl[i] - a_curr * xl[i] - b
        g_curr += xl[i] * w * r + a_curr * sx2[i] * w * w * r * r
    
    for _ in range(50):
        dg = g_curr - g_prev
        if abs(dg) < 1e-30:
            break
        a_new = a_curr - g_curr * (a_curr - a_prev) / dg
        a_prev = a_curr
        g_prev = g_curr
        a_curr = a_new
        
        a2 = a_curr * a_curr
        sw = swx = swy = 0.0
        for i in range(n):
            w = 1.0 / (sy2[i] + a2 * sx2[i])
            sw += w; swx += w * xl[i]; swy += w * yl[i]
        b = (swy - a_curr * swx) / sw
        g_curr = 0.0
        for i in range(n):
            w = 1.0 / (sy2[i] + a2 * sx2[i])
            r = yl[i] - a_curr * xl[i] - b
            g_curr += xl[i] * w * r + a_curr * sx2[i] * w * w * r * r
        
        if abs(a_curr - a_prev) < 1e-13 * (1.0 + abs(a_curr)):
            break
    
    a2 = a_curr * a_curr
    sw = swx = swy = 0.0
    for i in range(n):
        w = 1.0 / (sy2[i] + a2 * sx2[i])
        sw += w; swx += w * xl[i]; swy += w * yl[i]
    b_opt = (swy - a_curr * swx) / sw
    
    return a_curr, b_opt


class Solver:
    def __init__(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        sx = np.array([0.1, 0.1, 0.1])
        sy = np.array([0.1, 0.1, 0.1])
        _solve_odr_numba(x, y, sx, sy)
    
    def solve(self, problem, **kwargs):
        xl = problem["x"]
        n = len(xl)
        
        if n <= 50:
            # Pure Python for small n to avoid numpy overhead
            a_opt, b_opt = _solve_odr_python(xl, problem["y"], problem["sx"], problem["sy"])
        else:
            x = np.asarray(xl, dtype=np.float64)
            y = np.asarray(problem["y"], dtype=np.float64)
            sx = np.asarray(problem["sx"], dtype=np.float64)
            sy = np.asarray(problem["sy"], dtype=np.float64)
            a_opt, b_opt = _solve_odr_numba(x, y, sx, sy)
        
        return {"beta": [a_opt, b_opt]}