import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

@njit(cache=True, fastmath=True)
def compute_derivatives_numba(y, masses, softening_sq, num_bodies, n3):
    positions = y[:n3].reshape((num_bodies, 3))
    velocities = y[n3:].reshape((num_bodies, 3))
    
    derivatives = np.empty(2 * n3)
    # Copy velocities directly
    for i in range(n3):
        derivatives[i] = y[n3 + i]
    
    accelerations = np.zeros((num_bodies, 3))
    
    # Use Newton's third law: F_ij = -F_ji to reduce computation by half
    for i in range(num_bodies):
        ax_i = 0.0
        ay_i = 0.0
        az_i = 0.0
        pi_x = positions[i, 0]
        pi_y = positions[i, 1]
        pi_z = positions[i, 2]
        
        for j in range(i + 1, num_bodies):
            r_ij_x = positions[j, 0] - pi_x
            r_ij_y = positions[j, 1] - pi_y
            r_ij_z = positions[j, 2] - pi_z
            dist_squared = r_ij_x*r_ij_x + r_ij_y*r_ij_y + r_ij_z*r_ij_z + softening_sq
            inv_dist_cubed = 1.0 / (dist_squared * np.sqrt(dist_squared))
            
            # Acceleration on i due to j
            factor_i = masses[j] * inv_dist_cubed
            ax_i += factor_i * r_ij_x
            ay_i += factor_i * r_ij_y
            az_i += factor_i * r_ij_z
            
            # Acceleration on j due to i (Newton's third law)
            factor_j = masses[i] * inv_dist_cubed
            accelerations[j, 0] -= factor_j * r_ij_x
            accelerations[j, 1] -= factor_j * r_ij_y
            accelerations[j, 2] -= factor_j * r_ij_z
        
        accelerations[i, 0] += ax_i
        accelerations[i, 1] += ay_i
        accelerations[i, 2] += az_i
    
    for i in range(num_bodies):
        derivatives[n3 + i*3] = accelerations[i, 0]
        derivatives[n3 + i*3 + 1] = accelerations[i, 1]
        derivatives[n3 + i*3 + 2] = accelerations[i, 2]
    
    return derivatives

class Solver:
    def __init__(self):
        # Pre-compile the numba function with dummy data
        dummy_y = np.zeros(12, dtype=np.float64)
        dummy_masses = np.ones(2, dtype=np.float64)
        _ = compute_derivatives_numba(dummy_y, dummy_masses, 1e-8, 2, 6)
    
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        masses = np.array(problem["masses"], dtype=np.float64)
        softening = float(problem["softening"])
        num_bodies = int(problem["num_bodies"])
        n3 = num_bodies * 3
        softening_sq = softening * softening
        
        def rhs(t, y):
            return compute_derivatives_numba(y, masses, softening_sq, num_bodies, n3)
        
        sol = solve_ivp(rhs, [t0, t1], y0, method='RK45', rtol=1e-8, atol=1e-8)
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")