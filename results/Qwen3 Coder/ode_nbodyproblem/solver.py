from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
import numba

@numba.jit(nopython=True, fastmath=True)
def compute_accelerations_symmetric(positions, masses, softening, num_bodies):
    """Compute accelerations using symmetric force calculations to reduce computations."""
    accelerations = np.zeros_like(positions)
    
    # Precompute softening squared
    softening_sq = softening * softening
    
    # Use symmetry: force on i from j is equal and opposite to force on j from i
    for i in range(num_bodies):
        for j in range(i + 1, num_bodies):
            # Calculate vector from body i to body j
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            
            # Calculate squared distance with softening
            dist_sq = dx*dx + dy*dy + dz*dz + softening_sq
            
            # Calculate force factor: G * m_j / r_ij^3 (G=1)
            # Force on i from j: factor * (dx, dy, dz)
            # Force on j from i: factor * (-dx, -dy, -dz)
            factor = 1.0 / (dist_sq * np.sqrt(dist_sq))
            
            # Acceleration on i from j
            acc_i_x = factor * masses[j] * dx
            acc_i_y = factor * masses[j] * dy
            acc_i_z = factor * masses[j] * dz
            
            # Acceleration on j from i (opposite direction)
            acc_j_x = -factor * masses[i] * dx
            acc_j_y = -factor * masses[i] * dy
            acc_j_z = -factor * masses[i] * dz
            
            # Add to total accelerations
            accelerations[i, 0] += acc_i_x
            accelerations[i, 1] += acc_i_y
            accelerations[i, 2] += acc_i_z
            
            accelerations[j, 0] += acc_j_x
            accelerations[j, 1] += acc_j_y
            accelerations[j, 2] += acc_j_z
    
    return accelerations
    
    return accelerations

class Solver:
    def solve(self, problem):
        # Extract parameters
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        masses = np.array(problem["masses"])
        softening = problem["softening"]
        num_bodies = problem["num_bodies"]
        
        def nbody_derivatives(t, y):
            # Reshape state into positions and velocities
            positions = y[:num_bodies * 3].reshape(num_bodies, 3)
            velocities = y[num_bodies * 3:].reshape(num_bodies, 3)
            
            # Position derivatives = velocities
            dp_dt = velocities.reshape(-1)
            
            # Compute accelerations using symmetric force calculations
            accelerations = compute_accelerations_symmetric(positions, masses, softening, num_bodies)
            
            # Velocity derivatives = accelerations
            dv_dt = accelerations.reshape(-1)
            
            # Combine derivatives
            derivatives = np.concatenate([dp_dt, dv_dt])
            return derivatives
        
        # Solve the ODE with optimized parameters
        sol = solve_ivp(
            nbody_derivatives,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            t_eval=[t1],      # Only evaluate at final time
            dense_output=False,
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")
            # Compute accelerations using symmetric approach
            accelerations = compute_accelerations_symmetric(positions, masses, softening, num_bodies)
            
            # Velocity derivatives = accelerations
            dv_dt = accelerations.flatten()
            
            # Combine derivatives
            derivatives = np.concatenate([dp_dt, dv_dt])
            return derivatives
        
        # Solve the ODE with optimized parameters
        sol = solve_ivp(
            nbody_derivatives,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            t_eval=None,
            dense_output=False,
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")