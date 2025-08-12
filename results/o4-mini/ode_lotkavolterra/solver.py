import numba
from typing import Any

@numba.njit(fastmath=True)
def _dv_rhs(x, y, alpha, beta, delta, gamma):
    return alpha * x - beta * x * y, delta * x * y - gamma * y

@numba.njit(fastmath=True)
def _solve_adaptive(t0, t1, x, y, alpha, beta, delta, gamma):
    t = t0
    dt = (t1 - t0) / 100.0
    rtol = 1e-8
    atol = 1e-10
    safety = 0.9
    min_scale = 0.2
    max_scale = 5.0
    max_steps = 1000000
    steps = 0
    while t < t1 and steps < max_steps:
        # adjust last step
        if t + dt > t1:
            dt = t1 - t
        # compute RK stages
        k1x, k1y = _dv_rhs(x, y, alpha, beta, delta, gamma)
        x2 = x + dt * (1/5) * k1x
        y2 = y + dt * (1/5) * k1y
        k2x, k2y = _dv_rhs(x2, y2, alpha, beta, delta, gamma)
        x3 = x + dt * (3/40 * k1x + 9/40 * k2x)
        y3 = y + dt * (3/40 * k1y + 9/40 * k2y)
        k3x, k3y = _dv_rhs(x3, y3, alpha, beta, delta, gamma)
        x4 = x + dt * (44/45 * k1x - 56/15 * k2x + 32/9 * k3x)
        y4 = y + dt * (44/45 * k1y - 56/15 * k2y + 32/9 * k3y)
        k4x, k4y = _dv_rhs(x4, y4, alpha, beta, delta, gamma)
        x5 = x + dt * (19372/6561 * k1x - 25360/2187 * k2x + 64448/6561 * k3x - 212/729 * k4x)
        y5 = y + dt * (19372/6561 * k1y - 25360/2187 * k2y + 64448/6561 * k3y - 212/729 * k4y)
        k5x, k5y = _dv_rhs(x5, y5, alpha, beta, delta, gamma)
        x6 = x + dt * (9017/3168 * k1x - 355/33 * k2x + 46732/5247 * k3x + 49/176 * k4x - 5103/18656 * k5x)
        y6 = y + dt * (9017/3168 * k1y - 355/33 * k2y + 46732/5247 * k3y + 49/176 * k4y - 5103/18656 * k5y)
        k6x, k6y = _dv_rhs(x6, y6, alpha, beta, delta, gamma)
        x7 = x + dt * (35/384 * k1x + 500/1113 * k3x + 125/192 * k4x - 2187/6784 * k5x + 11/84 * k6x)
        y7 = y + dt * (35/384 * k1y + 500/1113 * k3y + 125/192 * k4y - 2187/6784 * k5y + 11/84 * k6y)
        k7x, k7y = _dv_rhs(x7, y7, alpha, beta, delta, gamma)
        # 5th order solution
        x5th = x + dt * (35/384 * k1x + 500/1113 * k3x + 125/192 * k4x - 2187/6784 * k5x + 11/84 * k6x)
        y5th = y + dt * (35/384 * k1y + 500/1113 * k3y + 125/192 * k4y - 2187/6784 * k5y + 11/84 * k6y)
        # 4th order solution
        x4th = x + dt * (5179/57600 * k1x + 7571/16695 * k3x + 393/640 * k4x - 92097/339200 * k5x + 187/2100 * k6x + 1/40 * k7x)
        y4th = y + dt * (5179/57600 * k1y + 7571/16695 * k3y + 393/640 * k4y - 92097/339200 * k5y + 187/2100 * k6y + 1/40 * k7y)
        # error estimate
        err_x = x5th - x4th
        err_y = y5th - y4th
        # normalized error
        tol_x = atol + rtol * abs(x5th)
        tol_y = atol + rtol * abs(y5th)
        err_norm = max(abs(err_x) / tol_x, abs(err_y) / tol_y)
        # accept or reject
        if err_norm <= 1.0:
            t += dt
            x = x5th
            y = y5th
        # adapt step
        dt = dt * min(max_scale, max(min_scale, safety * err_norm ** (-0.2)))
        steps += 1
    return x, y

class Solver:
    def __init__(self):
        # Warm up JIT
        _solve_adaptive(0.0, 1.0, 1.0, 1.0, 1.1, 0.4, 0.1, 0.4)

    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        t0 = problem["t0"]
        t1 = problem["t1"]
        x0 = float(problem["y0"][0])
        y0 = float(problem["y0"][1])
        params = problem["params"]
        x, y = _solve_adaptive(t0, t1, x0, y0,
                               params["alpha"], params["beta"],
                               params["delta"], params["gamma"])
        # clip negatives
        if x < 0.0:
            x = 0.0
        if y < 0.0:
            y = 0.0
        return [x, y]