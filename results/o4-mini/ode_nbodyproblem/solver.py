import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

@njit(cache=True, fastmath=True)
def compute_derivatives(y, masses, softening, N):
    # y: [pos(3N), vel(3N)]
    r = y[:3*N].reshape((N,3))
    v = y[3*N:].reshape((N,3))
    acc = np.zeros((N,3), dtype=np.float64)
    for i in range(N):
        xi = r[i,0]; yi = r[i,1]; zi = r[i,2]
        for j in range(N):
            if i != j:
                dx = r[j,0] - xi
                dy = r[j,1] - yi
                dz = r[j,2] - zi
                dist2 = dx*dx + dy*dy + dz*dz + softening*softening
                inv_dist = 1.0/np.sqrt(dist2)
                inv_dist3 = inv_dist * inv_dist * inv_dist
                f = masses[j] * inv_dist3
                acc[i,0] += f * dx
                acc[i,1] += f * dy
                acc[i,2] += f * dz
    dy = np.empty(6*N, dtype=np.float64)
    # position derivatives = velocities
    for i in range(3*N):
        dy[i] = y[3*N + i]
    # velocity derivatives = accelerations
    flat_acc = acc.reshape(3*N)
    for i in range(3*N):
        dy[3*N + i] = flat_acc[i]
    return dy

# Warm up Numba compilation (excluded from runtime)
_dummy_y = np.zeros(6, dtype=np.float64)
_dummy_m = np.ones(1, dtype=np.float64)
compute_derivatives(_dummy_y, _dummy_m, 0.0, 1)

class Solver:
    def solve(self, problem, **kwargs) -> list:
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        N = int(problem["num_bodies"])
        y0 = np.array(problem["y0"], dtype=np.float64)
        masses = np.array(problem["masses"], dtype=np.float64)
        softening = float(problem["softening"])

        def fun(t, y):
            return compute_derivatives(y, masses, softening, N)

        sol = solve_ivp(fun, (t0, t1), y0, method="RK45", rtol=1e-8, atol=1e-8)
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        y_final = sol.y[:, -1]
        return y_final.tolist()