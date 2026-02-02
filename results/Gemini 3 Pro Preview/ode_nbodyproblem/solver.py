import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

@njit(fastmath=True)
def nbody_derivs(t, y, masses, softening, num_bodies):
    # y is [x1, y1, z1, ..., vx1, vy1, vz1, ...]
    # positions: y[:3*num_bodies]
    # velocities: y[3*num_bodies:]
    
    dydt = np.empty(6*num_bodies, dtype=np.float64)
    
    # The first half of derivatives are velocities
    dydt[:3*num_bodies] = y[3*num_bodies:]
    
    # The second half are accelerations
    # We can access the acceleration part of dydt directly
    # But we need to initialize it to 0
    acc_offset = 3 * num_bodies
    dydt[acc_offset:] = 0.0
    
    pos = y
    
    soft_sq = softening * softening
    
    for i in range(num_bodies):
        idx_i = i * 3
        xi = pos[idx_i]
        yi = pos[idx_i+1]
        zi = pos[idx_i+2]
        
        for j in range(i + 1, num_bodies):
            idx_j = j * 3
            dx = pos[idx_j] - xi
            dy = pos[idx_j+1] - yi
            dz = pos[idx_j+2] - zi
            
            dist_sq = dx*dx + dy*dy + dz*dz + soft_sq
            inv_dist3 = 1.0 / (dist_sq * np.sqrt(dist_sq))
            
            # Force on i from j
            mj_fac = masses[j] * inv_dist3
            dydt[acc_offset + idx_i]   += mj_fac * dx
            dydt[acc_offset + idx_i+1] += mj_fac * dy
            dydt[acc_offset + idx_i+2] += mj_fac * dz
            
            # Force on j from i
            mi_fac = masses[i] * inv_dist3
            dydt[acc_offset + idx_j]   -= mi_fac * dx
            dydt[acc_offset + idx_j+1] -= mi_fac * dy
            dydt[acc_offset + idx_j+2] -= mi_fac * dz
            
    return dydt

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = problem["t0"]
        t1 = problem["t1"]
        masses = np.array(problem["masses"], dtype=np.float64)
        softening = problem["softening"]
        num_bodies = problem["num_bodies"]
        
        sol = solve_ivp(
            nbody_derivs,
            (t0, t1),
            y0,
            method='RK45',
            rtol=1e-8,
            atol=1e-8,
            args=(masses, softening, num_bodies)
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")