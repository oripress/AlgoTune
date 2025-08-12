import numpy as np
from scipy.integrate import solve_ivp
import numba as nb

@nb.njit(fastmath=True)
def compute_acc(pos, masses, softening):
    N = pos.shape[0]
    acc = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        ax = 0.0
        ay = 0.0
        az = 0.0
        xi = pos[i, 0]
        yi = pos[i, 1]
        zi = pos[i, 2]
        for j in range(N):
            if i == j:
                continue
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            dz = pos[j, 2] - zi
            dist2 = dx * dx + dy * dy + dz * dz + softening * softening
            inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
            m = masses[j]
            ax += m * dx * inv_dist3
            ay += m * dy * inv_dist3
            az += m * dz * inv_dist3
        acc[i, 0] = ax
        acc[i, 1] = ay
        acc[i, 2] = az
    return acc
class Solver:
    def solve(self, problem):
        """
        Solve the N‑body gravitational problem using a high‑accuracy RK45 integrator.
        The implementation uses a fully vectorized acceleration computation
        (no explicit Python loops) for speed.
        """
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        masses = np.asarray(problem["masses"], dtype=float)
        softening = float(problem["softening"])
        N = int(problem["num_bodies"])
        dim = 3  # 3‑D positions
        expected_len = 2 * N * dim
        if y0.size < expected_len:
            y0 = np.concatenate([y0, np.zeros(expected_len - y0.size, dtype=float)])
        elif y0.size > expected_len:
            y0 = y0[:expected_len]
        # Define the ODE system for the N‑body problem
        def nbody_ode(t, y):
            # Split state vector into positions and velocities
            pos = y[: N * dim].reshape(N, dim)
            vel = y[N * dim :].reshape(N, dim)

            # Position derivatives are velocities
            dp_dt = vel.ravel()

            # Compute accelerations using the Numba‑accelerated function
            acc = compute_acc(pos, masses, softening)

            # Velocity derivatives are accelerations
            dv_dt = acc.ravel()

            # Return concatenated derivatives
            return np.concatenate([dp_dt, dv_dt])

        # Integrate the ODE from t0 to t1 using a high‑accuracy RK45 method
        sol = solve_ivp(
            nbody_ode,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        # Return final state as a flat list
        return sol.y[:, -1].tolist()

        # Return final state (positions and velocities) as a flat list
        return sol.y[:, -1].tolist()