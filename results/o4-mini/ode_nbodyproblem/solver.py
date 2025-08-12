import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

@njit(nogil=True, fastmath=True)
def compute_acc_numba(pos, masses, soft2):
    N = masses.shape[0]
    acc = np.zeros(N * 3, dtype=np.float64)
    for i in range(N):
        xi = pos[3 * i]
        yi = pos[3 * i + 1]
        zi = pos[3 * i + 2]
        axi = 0.0
        ayi = 0.0
        azi = 0.0
        for j in range(N):
            if i != j:
                dx = pos[3 * j]     - xi
                dy = pos[3 * j + 1] - yi
                dz = pos[3 * j + 2] - zi
                dist2 = dx*dx + dy*dy + dz*dz + soft2
                invd3 = 1.0 / (dist2 * np.sqrt(dist2))
                f = masses[j] * invd3
                axi += f * dx
                ayi += f * dy
                azi += f * dz
        acc[3 * i]     = axi
        acc[3 * i + 1] = ayi
        acc[3 * i + 2] = azi
    return acc

# Trigger JIT compile of acceleration routine (not counted in solve runtime)
_dummy_pos = np.zeros(3, dtype=np.float64)
_dummy_masses = np.ones(1, dtype=np.float64)
_compute_dummy = compute_acc_numba(_dummy_pos, _dummy_masses, 0.0)

class Solver:
    def solve(self, problem, **kwargs):
        # Unpack problem data
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        masses = np.array(problem["masses"], dtype=np.float64)
        soft2 = float(problem["softening"])**2
        N = int(problem["num_bodies"])

        # ODE derivative using Numba accel for forces
        def nbody(t, y):
            deriv = np.empty_like(y)
            # positions derivative = velocities
            deriv[:3*N] = y[3*N:]
            # velocities derivative = accelerations
            deriv[3*N:] = compute_acc_numba(y[:3*N], masses, soft2)
            return deriv

        # Integrate with adaptive RK45 solver
        sol = solve_ivp(
            nbody,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        return sol.y[:, -1].tolist()