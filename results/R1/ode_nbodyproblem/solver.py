import numpy as np
from scipy.integrate import solve_ivp
from numba import jit

@jit(nopython=True, fastmath=True)
def compute_accelerations(positions, masses, softening):
    num_bodies = positions.shape[0]
    accelerations = np.zeros((num_bodies, 3))
    soft_sq = softening**2
    
    # Use symmetric force calculations (Newton's 3rd law)
    for i in range(num_bodies):
        for j in range(i+1, num_bodies):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            dist_sq = dx*dx + dy*dy + dz*dz + soft_sq
            
            dist = np.sqrt(dist_sq)
            inv_dist_cube = 1.0 / (dist_sq * dist)
            
            fx = dx * inv_dist_cube
            fy = dy * inv_dist_cube
            fz = dz * inv_dist_cube
            
            mj = masses[j]
            mi = masses[i]
            
            accelerations[i, 0] += fx * mj
            accelerations[i, 1] += fy * mj
            accelerations[i, 2] += fz * mj
            accelerations[j, 0] -= fx * mi
            accelerations[j, 1] -= fy * mi
            accelerations[j, 2] -= fz * mi
            
    return accelerations

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        masses = np.array(problem["masses"])
        softening = problem["softening"]
        num_bodies = problem["num_bodies"]
        
        def nbody_ode(t, y):
            positions = y[:3*num_bodies].reshape((num_bodies, 3))
            velocities = y[3*num_bodies:].reshape((num_bodies, 3))
            
            dp_dt = velocities.ravel()
            acc = compute_accelerations(positions, masses, softening)
            dv_dt = acc.ravel()
            
            return np.concatenate((dp_dt, dv_dt))
        
        # Use the same solver as the reference implementation
        sol = solve_ivp(
            nbody_ode,
            (t0, t1),
            y0,
            method='RK45',
            rtol=1e-8,
            atol=1e-8,
            dense_output=False
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")