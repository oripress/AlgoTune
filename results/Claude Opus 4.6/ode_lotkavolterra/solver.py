import numba as nb
import numpy as np

@nb.njit(cache=True)
def _integrate(t0, t1, x0, y0, alpha, beta, delta, gamma, rtol, atol):
    a21 = 0.2
    a31 = 0.075
    a32 = 0.225
    a41 = 44.0 / 45.0
    a42 = -56.0 / 15.0
    a43 = 32.0 / 9.0
    a51 = 19372.0 / 6561.0
    a52 = -25360.0 / 2187.0
    a53 = 64448.0 / 6561.0
    a54 = -212.0 / 729.0
    a61 = 9017.0 / 3168.0
    a62 = -355.0 / 33.0
    a63 = 46732.0 / 5247.0
    a64 = 49.0 / 176.0
    a65 = -5103.0 / 18656.0
    b1 = 35.0 / 384.0
    b3 = 500.0 / 1113.0
    b4 = 125.0 / 192.0
    b5 = -2187.0 / 6784.0
    b6 = 11.0 / 84.0
    e1 = 71.0 / 57600.0
    e3 = -71.0 / 16695.0
    e4 = 71.0 / 1920.0
    e5 = -17253.0 / 339200.0
    e6 = 22.0 / 525.0
    e7 = -1.0 / 40.0

    t = t0
    x = x0
    y = y0

    f0x = alpha * x - beta * x * y
    f0y = delta * x * y - gamma * y
    d0 = max(abs(x), abs(y))
    d1 = max(abs(f0x), abs(f0y))
    if d0 < 1e-5 or d1 < 1e-5:
        h = 1e-6
    else:
        h = 0.01 * d0 / d1
    h = min(h, t1 - t0)

    while t < t1:
        if t + h > t1:
            h = t1 - t
        if h <= 0.0:
            break

        k1x = alpha * x - beta * x * y
        k1y = delta * x * y - gamma * y

        tx = x + h * a21 * k1x
        ty = y + h * a21 * k1y
        k2x = alpha * tx - beta * tx * ty
        k2y = delta * tx * ty - gamma * ty

        tx = x + h * (a31 * k1x + a32 * k2x)
        ty = y + h * (a31 * k1y + a32 * k2y)
        k3x = alpha * tx - beta * tx * ty
        k3y = delta * tx * ty - gamma * ty

        tx = x + h * (a41 * k1x + a42 * k2x + a43 * k3x)
        ty = y + h * (a41 * k1y + a42 * k2y + a43 * k3y)
        k4x = alpha * tx - beta * tx * ty
        k4y = delta * tx * ty - gamma * ty

        tx = x + h * (a51 * k1x + a52 * k2x + a53 * k3x + a54 * k4x)
        ty = y + h * (a51 * k1y + a52 * k2y + a53 * k3y + a54 * k4y)
        k5x = alpha * tx - beta * tx * ty
        k5y = delta * tx * ty - gamma * ty

        tx = x + h * (a61 * k1x + a62 * k2x + a63 * k3x + a64 * k4x + a65 * k5x)
        ty = y + h * (a61 * k1y + a62 * k2y + a63 * k3y + a64 * k4y + a65 * k5y)
        k6x = alpha * tx - beta * tx * ty
        k6y = delta * tx * ty - gamma * ty

        xn = x + h * (b1 * k1x + b3 * k3x + b4 * k4x + b5 * k5x + b6 * k6x)
        yn = y + h * (b1 * k1y + b3 * k3y + b4 * k4y + b5 * k5y + b6 * k6y)

        k7x = alpha * xn - beta * xn * yn
        k7y = delta * xn * yn - gamma * yn

        ex = h * (e1 * k1x + e3 * k3x + e4 * k4x + e5 * k5x + e6 * k6x + e7 * k7x)
        ey = h * (e1 * k1y + e3 * k3y + e4 * k4y + e5 * k5y + e6 * k6y + e7 * k7y)

        sx = atol + rtol * max(abs(x), abs(xn))
        sy = atol + rtol * max(abs(y), abs(yn))
        err = ((ex / sx) ** 2 + (ey / sy) ** 2) ** 0.5 * 0.7071067811865476

        if err <= 1.0:
            t += h
            x = xn
            y = yn
            if err < 1e-10:
                h *= 5.0
            else:
                fac = 0.9 * err ** (-0.2)
                if fac > 5.0:
                    fac = 5.0
                elif fac < 0.2:
                    fac = 0.2
                h *= fac
        else:
            fac = 0.9 * err ** (-0.2)
            if fac < 0.2:
                fac = 0.2
            h *= fac

    return x, y

class Solver:
    def __init__(self):
        _integrate(0.0, 1.0, 10.0, 5.0, 1.1, 0.4, 0.1, 0.4, 1e-8, 1e-8)

    def solve(self, problem, **kwargs):
        y0 = problem["y0"]
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        p = problem["params"]
        x, y = _integrate(
            t0, t1, float(y0[0]), float(y0[1]),
            float(p["alpha"]), float(p["beta"]),
            float(p["delta"]), float(p["gamma"]),
            1e-8, 1e-8
        )
        return [x, y]