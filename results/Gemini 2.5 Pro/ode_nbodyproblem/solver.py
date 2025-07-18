from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Solves the N-body gravitational problem using a vectorized approach.
        """
        y0 = np.array(problem["y0"])
        t_span = [problem["t0"], problem["t1"]]
        masses = np.array(problem["masses"])
        softening = problem["softening"]
        num_bodies = problem["num_bodies"]

        def nbody_ode(t, y):
            # Reshape the 1D state vector into 2D arrays for positions and velocities
            positions = y[:num_bodies * 3].reshape(num_bodies, 3)
            velocities = y[num_bodies * 3:].reshape(num_bodies, 3)

            # --- Vectorized calculation of accelerations ---

            # Calculate all pairwise position difference vectors
            # r_diffs[i, j] = positions[j] - positions[i]
            r_diffs = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]

            # Calculate squared distances with softening
            # dist_sq[i, j] = ||positions[j] - positions[i]||^2 + softening^2
            dist_sq = np.sum(r_diffs**2, axis=2) + softening**2
            
            # Calculate 1 / (distance^3) for the force law
            inv_dist_cubed = dist_sq**(-1.5)
            
            # Set self-interaction terms to zero to avoid division by zero and self-force
            np.fill_diagonal(inv_dist_cubed, 0.)

            # Calculate acceleration factors: G*m_j / dist^3 (with G=1)
            # This broadcasts masses across each row
            accel_factors = masses[np.newaxis, :] * inv_dist_cubed

            # Calculate accelerations by summing forces from all other bodies
            # accel = sum_j (accel_factors[i,j] * r_diffs[i,j])
            accelerations = np.sum(accel_factors[:, :, np.newaxis] * r_diffs, axis=1)

            # The derivative of position is velocity
            dp_dt = velocities.flatten()
            # The derivative of velocity is acceleration
            dv_dt = accelerations.flatten()

            # Combine into a single derivative vector
            return np.concatenate((dp_dt, dv_dt))

        # Use a standard Runge-Kutta method for integration
        sol = solve_ivp(
            nbody_ode,
            t_span,
            y0,
            method='RK45',
            rtol=1e-8,
            atol=1e-8,
        )

        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            # Raise an error if the integration fails to converge
            raise RuntimeError(f"Solver failed: {sol.message}")