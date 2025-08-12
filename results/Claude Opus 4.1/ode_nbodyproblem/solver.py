import numpy as np
from scipy.integrate import solve_ivp
from numba import jit
from typing import Any

@jit(nopython=True, cache=True, fastmath=True)
def compute_accelerations(positions, masses, softening, num_bodies):
    """Compute gravitational accelerations for all bodies."""
    accelerations = np.zeros((num_bodies, 3))
    
    for i in range(num_bodies):
        for j in range(i + 1, num_bodies):
            # Vector from body i to body j
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            
            # Squared distance with softening
            dist_squared = dx*dx + dy*dy + dz*dz + softening*softening
            dist_cubed = dist_squared * np.sqrt(dist_squared)
            
            # Mutual forces (Newton's third law)
            force_i = masses[j] / dist_cubed
            force_j = masses[i] / dist_cubed
            
            # Apply forces
            accelerations[i, 0] += force_i * dx
            accelerations[i, 1] += force_i * dy
            accelerations[i, 2] += force_i * dz
            
            accelerations[j, 0] -= force_j * dx
            accelerations[j, 1] -= force_j * dy
            accelerations[j, 2] -= force_j * dz
    
    return accelerations

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Solve the N-body gravitational problem."""
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        masses = np.array(problem["masses"], dtype=np.float64)
        softening = problem["softening"]
        num_bodies = problem["num_bodies"]
        
        def nbodyproblem(t, y):
            # Split state into positions and velocities
            positions = y[:num_bodies * 3].reshape(num_bodies, 3)
            velocities = y[num_bodies * 3:].reshape(num_bodies, 3)
            
            # Compute accelerations using JIT-compiled function
            accelerations = compute_accelerations(positions, masses, softening, num_bodies)
            
            # Return derivatives: [velocities, accelerations]
            derivatives = np.empty_like(y)
            derivatives[:num_bodies * 3] = velocities.ravel()
            derivatives[num_bodies * 3:] = accelerations.ravel()
            return derivatives
        
        # Solve with RK45
        sol = solve_ivp(
            nbodyproblem,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")