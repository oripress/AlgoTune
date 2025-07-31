import numba
import math
@numba.njit
def _integrate(x0, y0, alpha, beta, delta, gamma, total_time):
    steps = int(total_time * 50.0) + 1
    dt = total_time / steps
    half_dt = dt * 0.5
    sixth_dt = dt / 6.0
    x = x0
    y = y0
    for i in range(steps):
        # k1
        d1x = x * (alpha - beta * y)
        d1y = y * (delta * x - gamma)
        # k2
        xt = x + half_dt * d1x
        yt = y + half_dt * d1y
        d2x = xt * (alpha - beta * yt)
        d2y = yt * (delta * xt - gamma)
        # k3
        xt = x + half_dt * d2x
        yt = y + half_dt * d2y
        d3x = xt * (alpha - beta * yt)
        d3y = yt * (delta * xt - gamma)
        # k4
        xt = x + dt * d3x
        yt = y + dt * d3y
        d4x = xt * (alpha - beta * yt)
        d4y = yt * (delta * xt - gamma)
        # update
        x += sixth_dt * (d1x + 2.0 * d2x + 2.0 * d3x + d4x)
        y += sixth_dt * (d1y + 2.0 * d2y + 2.0 * d3y + d4y)
    return x, y

class Solver:
    def solve(self, problem, **kwargs):
        # Unpack times
        t0 = problem["t0"]
        t1 = problem["t1"]
        total_time = t1 - t0
        # Initial populations
        x0 = float(problem["y0"][0])
        y0 = float(problem["y0"][1])
        if total_time <= 0:
            return [x0, y0]
        # Parameters
        p = problem["params"]
        alpha = p["alpha"]; beta = p["beta"]
        delta = p["delta"]; gamma = p["gamma"]
        # Integrate via compiled RK4
        x_final, y_final = _integrate(x0, y0, alpha, beta, delta, gamma, total_time)
        return [x_final, y_final]