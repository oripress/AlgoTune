import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict, **kwargs):
        """Solve the N-body gravitational problem."""
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        masses = np.array(problem["masses"])
        softening = problem["softening"]
        num_bodies = problem["num_bodies"]
        
        def nbodyproblem(t, y):
            # Reshape state into positions and velocities
            positions = y[:num_bodies * 3].reshape(num_bodies, 3)
            velocities = y[num_bodies * 3:].reshape(num_bodies, 3)
            
            # Position derivatives = velocities
            dp_dt = velocities.reshape(-1)
            
            # Vectorized computation of accelerations
            # Compute all pairwise displacement vectors
            # r[i,j] = vector from body i to body j
            r = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # shape: (n, n, 3)
            
            # Compute distances squared with softening
            dist_squared = np.sum(r**2, axis=2) + softening**2  # shape: (n, n)
            
            # Avoid division by zero on diagonal
            np.fill_diagonal(dist_squared, 1.0)
            
            # Compute force factors
            factors = masses[np.newaxis, :] / (dist_squared * np.sqrt(dist_squared))  # shape: (n, n)
            np.fill_diagonal(factors, 0.0)  # Zero out self-interactions
            
            # Compute accelerations
            accelerations = np.sum(factors[:, :, np.newaxis] * r, axis=1)  # shape: (n, 3)
            
            # Velocity derivatives = accelerations
            dv_dt = accelerations.reshape(-1)
            
            return np.concatenate([dp_dt, dv_dt])
        
        # Solve with tighter tolerances to match reference accuracy
        sol = solve_ivp(
            nbodyproblem,
            [t0, t1],
            y0,
            method='RK45',
            rtol=1e-8,
            atol=1e-8
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")