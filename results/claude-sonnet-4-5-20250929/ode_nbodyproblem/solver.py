import numpy as np
from scipy.integrate import solve_ivp
from numba import jit

@jit(nopython=True, cache=True, fastmath=True)
def compute_derivatives_numba(y, masses, softening, num_bodies):
    """Numba-compiled derivative computation with everything inlined"""
    # Extract positions and velocities
    positions = y[:num_bodies * 3].reshape(num_bodies, 3)
    
    # Position derivatives = velocities
    dp_dt = y[num_bodies * 3:]
    
    # Compute accelerations
    accelerations = np.zeros(num_bodies * 3)
    
    for i in range(num_bodies):
        ax, ay, az = 0.0, 0.0, 0.0
        px, py, pz = positions[i, 0], positions[i, 1], positions[i, 2]
        
        for j in range(num_bodies):
            if i != j:
                # Vector from body i to body j
                dx = positions[j, 0] - px
                dy = positions[j, 1] - py
                dz = positions[j, 2] - pz
                
                # Squared distance with softening
                dist_squared = dx*dx + dy*dy + dz*dz + softening*softening
                
                # Force factor
                inv_dist = 1.0 / np.sqrt(dist_squared)
                factor = masses[j] * inv_dist * inv_dist * inv_dist
                
                # Accumulate acceleration contribution
                ax += factor * dx
                ay += factor * dy
                az += factor * dz
        
        accelerations[i * 3] = ax
        accelerations[i * 3 + 1] = ay
        accelerations[i * 3 + 2] = az
    
    # Combine derivatives
    derivatives = np.empty(len(y))
    derivatives[:num_bodies * 3] = dp_dt
    derivatives[num_bodies * 3:] = accelerations
    
    return derivatives

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        masses = np.array(problem["masses"])
        softening = problem["softening"]
        num_bodies = problem["num_bodies"]
        
        def nbodyproblem(t, y):
            return compute_derivatives_numba(y, masses, softening, num_bodies)
        
        sol = solve_ivp(
            nbodyproblem,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")