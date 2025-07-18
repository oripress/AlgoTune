import numpy as np
import numba

@numba.njit(cache=True, fastmath=True)
def _integrate(t0, t1, y0arr, A, B):
    nt = t1 - t0
    nsteps = int(100.0 * nt)
    if nsteps < 1:
        nsteps = 1
    dt = nt / nsteps
    x = y0arr[0]; y = y0arr[1]
    for _ in range(nsteps):
        # Compute slopes
        k1x = A + x*x*y - (B + 1.0) * x
        k1y = B*x - x*x*y
        xt = x + 0.5 * dt * k1x
        yt = y + 0.5 * dt * k1y
        k2x = A + xt*xt*yt - (B + 1.0) * xt
        k2y = B*xt - xt*xt*yt
        xt2 = x + 0.5 * dt * k2x
        yt2 = y + 0.5 * dt * k2y
        k3x = A + xt2*xt2*yt2 - (B + 1.0) * xt2
        k3y = B*xt2 - xt2*xt2*yt2
        xt3 = x + dt * k3x
        yt3 = y + dt * k3y
        k4x = A + xt3*xt3*yt3 - (B + 1.0) * xt3
        k4y = B*xt3 - xt3*xt3*yt3
        # Update state
        x += dt * (k1x + 2.0*k2x + 2.0*k3x + k4x) / 6.0
        y += dt * (k1y + 2.0*k2y + 2.0*k3y + k4y) / 6.0
    return x, y

# Warm-up compilation (not counted towards solve runtime)
_integrate(0.0, 0.0, np.zeros(2, dtype=np.float64), 0.0, 0.0)

class Solver:
    def solve(self, problem, **kwargs):
        t0 = float(problem.get("t0", 0.0))
        t1 = float(problem.get("t1", 0.0))
        y0_list = problem.get("y0", [0.0, 0.0])
        y0arr = np.array(y0_list, dtype=np.float64)
        params = problem.get("params", {})
        A = float(params.get("A", 0.0))
        B = float(params.get("B", 0.0))
        # Quick return if no integration needed
        if t1 <= t0:
            return y0arr.tolist()
        # Integrate with RK4
        x_final, y_final = _integrate(t0, t1, y0arr, A, B)
        return [x_final, y_final]