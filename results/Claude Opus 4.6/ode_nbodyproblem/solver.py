import numpy as np
import numba


@numba.njit(cache=True, fastmath=True)
def compute_derivs(y, masses, softening_sq, num_bodies):
    n3 = num_bodies * 3
    n = 2 * n3
    deriv = np.empty(n)

    for i in range(n3):
        deriv[i] = y[n3 + i]

    for i in range(n3, n):
        deriv[i] = 0.0

    for i in range(num_bodies):
        i3 = i * 3
        xi = y[i3]
        yi_c = y[i3 + 1]
        zi = y[i3 + 2]
        mi = masses[i]
        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in range(i + 1, num_bodies):
            j3 = j * 3
            rx = y[j3] - xi
            ry = y[j3 + 1] - yi_c
            rz = y[j3 + 2] - zi

            dist_sq = rx * rx + ry * ry + rz * rz + softening_sq
            inv_dist3 = 1.0 / (dist_sq * np.sqrt(dist_sq))

            fj = masses[j] * inv_dist3
            fi = mi * inv_dist3

            ax += fj * rx
            ay += fj * ry
            az += fj * rz

            deriv[n3 + j3] -= fi * rx
            deriv[n3 + j3 + 1] -= fi * ry
            deriv[n3 + j3 + 2] -= fi * rz

        deriv[n3 + i3] += ax
        deriv[n3 + i3 + 1] += ay
        deriv[n3 + i3 + 2] += az

    return deriv


@numba.njit(cache=True, fastmath=True)
def solve_rk45(y0, t0, t1, masses, softening_sq, num_bodies, rtol, atol):
    n = len(y0)
    y = y0.copy()
    t = t0

    # Dormand-Prince coefficients
    a21 = 0.2
    a31 = 3.0 / 40.0;       a32 = 9.0 / 40.0
    a41 = 44.0 / 45.0;      a42 = -56.0 / 15.0;     a43 = 32.0 / 9.0
    a51 = 19372.0 / 6561.0;  a52 = -25360.0 / 2187.0; a53 = 64448.0 / 6561.0;  a54 = -212.0 / 729.0
    a61 = 9017.0 / 3168.0;   a62 = -355.0 / 33.0;    a63 = 46732.0 / 5247.0;   a64 = 49.0 / 176.0;     a65 = -5103.0 / 18656.0

    b1 = 35.0 / 384.0; b3 = 500.0 / 1113.0; b4 = 125.0 / 192.0; b5 = -2187.0 / 6784.0; b6 = 11.0 / 84.0

    e1 = 71.0 / 57600.0;    e3 = -71.0 / 16695.0
    e4 = 71.0 / 1920.0;     e5 = -17253.0 / 339200.0
    e6 = 22.0 / 525.0;      e7 = -1.0 / 40.0

    k1 = compute_derivs(y, masses, softening_sq, num_bodies)

    # Initial step size estimation (Hairer's algorithm)
    d0_sq = 0.0
    d1_sq = 0.0
    for i in range(n):
        sc = atol + rtol * abs(y[i])
        d0_sq += (y[i] / sc) ** 2
        d1_sq += (k1[i] / sc) ** 2
    d0 = np.sqrt(d0_sq / n)
    d1 = np.sqrt(d1_sq / n)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    h0 = min(h0, t1 - t0)

    ytmp = np.empty(n)
    for i in range(n):
        ytmp[i] = y[i] + h0 * k1[i]
    k_e = compute_derivs(ytmp, masses, softening_sq, num_bodies)

    d2_sq = 0.0
    for i in range(n):
        sc = atol + rtol * abs(y[i])
        d2_sq += ((k_e[i] - k1[i]) / sc) ** 2
    d2 = np.sqrt(d2_sq / n) / h0

    if max(d1, d2) <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** 0.2

    h = min(100.0 * h0, h1, t1 - t0)

    for _ in range(10000000):
        if t >= t1 - 1e-14 * max(abs(t), abs(t1)):
            break

        h_step = min(h, t1 - t)
        if h_step <= 0.0:
            break

        # Stage 2
        for i in range(n):
            ytmp[i] = y[i] + h_step * a21 * k1[i]
        k2 = compute_derivs(ytmp, masses, softening_sq, num_bodies)

        # Stage 3
        for i in range(n):
            ytmp[i] = y[i] + h_step * (a31 * k1[i] + a32 * k2[i])
        k3 = compute_derivs(ytmp, masses, softening_sq, num_bodies)

        # Stage 4
        for i in range(n):
            ytmp[i] = y[i] + h_step * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i])
        k4 = compute_derivs(ytmp, masses, softening_sq, num_bodies)

        # Stage 5
        for i in range(n):
            ytmp[i] = y[i] + h_step * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i])
        k5 = compute_derivs(ytmp, masses, softening_sq, num_bodies)

        # Stage 6
        for i in range(n):
            ytmp[i] = y[i] + h_step * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i])
        k6 = compute_derivs(ytmp, masses, softening_sq, num_bodies)

        # 5th order solution
        for i in range(n):
            ytmp[i] = y[i] + h_step * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i])

        # k7 (FSAL property)
        k7 = compute_derivs(ytmp, masses, softening_sq, num_bodies)

        # Error estimation
        err_sq = 0.0
        for i in range(n):
            ei = h_step * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i])
            sci = atol + rtol * max(abs(y[i]), abs(ytmp[i]))
            err_sq += (ei / sci) ** 2
        err_norm = np.sqrt(err_sq / n)

        if err_norm <= 1.0:
            # Accept step
            t += h_step
            for i in range(n):
                y[i] = ytmp[i]
            k1 = k7

            if err_norm < 1e-30:
                factor = 10.0
            else:
                factor = min(10.0, max(0.2, 0.9 * err_norm ** (-0.2)))
            h = h_step * factor
        else:
            # Reject step
            factor = max(0.2, 0.9 * err_norm ** (-0.2))
            h = h_step * factor

    return y


class Solver:
    def __init__(self):
        # Trigger numba compilation (doesn't count towards runtime)
        test_y = np.zeros(12, dtype=np.float64)
        test_y[3] = 1.0
        test_y[10] = 1.0
        test_masses = np.array([1.0, 0.001], dtype=np.float64)
        _ = solve_rk45(test_y, 0.0, 0.01, test_masses, 1e-8, 2, 1e-8, 1e-8)

    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        masses = np.array(problem["masses"], dtype=np.float64)
        softening = float(problem["softening"])
        num_bodies = int(problem["num_bodies"])
        softening_sq = softening * softening

        result = solve_rk45(y0, t0, t1, masses, softening_sq, num_bodies, 1e-8, 1e-8)
        return result.tolist()