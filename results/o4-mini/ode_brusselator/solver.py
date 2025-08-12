import numpy as np
from numba import njit

@njit(fastmath=True)
def _integrate(t0, t1, y0, A, B):
    dt_total = t1 - t0
    # Use fixed step count proportional to total time, with a minimum
    N = int(100.0 * dt_total) + 1
    if N < 1000:
        N = 1000
    dt = dt_total / N
    dt2 = dt * 0.5
    dt6 = dt / 6.0
    ab = B + 1.0
    bB = B
    X = y0[0]
    Y = y0[1]
    for _ in range(N):
        X2 = X * X
        k1x = A + X2 * Y - ab * X
        k1y = bB * X - X2 * Y

        x_mid = X + dt2 * k1x
        y_mid = Y + dt2 * k1y
        Xmid2 = x_mid * x_mid
        k2x = A + Xmid2 * y_mid - ab * x_mid
        k2y = bB * x_mid - Xmid2 * y_mid

        x_mid = X + dt2 * k2x
        y_mid = Y + dt2 * k2y
        Xmid2 = x_mid * x_mid
        k3x = A + Xmid2 * y_mid - ab * x_mid
        k3y = bB * x_mid - Xmid2 * y_mid

        x_end = X + dt * k3x
        y_end = Y + dt * k3y
        Xend2 = x_end * x_end
        k4x = A + Xend2 * y_end - ab * x_end
        k4y = bB * x_end - Xend2 * y_end

        X += dt6 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        Y += dt6 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    return X, Y

class Solver:
    def __init__(self):
        # Force compile of the integrator (init time not counted)
        dummy = np.array([0.0, 0.0], dtype=np.float64)
        _integrate(0.0, 0.0, dummy, 0.0, 0.0)

    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        A = float(params["A"])
        B = float(params["B"])
        X, Y = _integrate(t0, t1, y0, A, B)
        return [X, Y]