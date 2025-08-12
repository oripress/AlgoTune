import numpy as np
from numba import jit
from scipy.integrate import solve_ivp

@jit(nopython=True, fastmath=True, cache=True, inline='always', boundscheck=False)
def compute_accelerations(positions, masses, softening, num_bodies):
    accelerations = np.zeros((num_bodies, 3))
    softening_sq = softening * softening
    
    for i in range(num_bodies):
        for j in range(num_bodies):
            if i != j:
                dx = positions[j,0] - positions[i,0]
                dy = positions[j,1] - positions[i,1]
                dz = positions[j,2] - positions[i,2]
                dist_sq = dx*dx + dy*dy + dz*dz + softening_sq
                # Combine sqrt and division for better performance
                factor = masses[j] / (dist_sq * np.sqrt(dist_sq))
                accelerations[i,0] += factor * dx
                accelerations[i,1] += factor * dy
                accelerations[i,2] += factor * dz
    return accelerations

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        masses = np.array(problem["masses"])
        softening = problem["softening"]
        num_bodies = problem["num_bodies"]
        
        def nbodyproblem(t, y):
            positions = y[: num_bodies * 3].reshape(num_bodies, 3)
            velocities = y[num_bodies * 3 :].reshape(num_bodies, 3)
            
            dp_dt = velocities.reshape(-1)
            accelerations = compute_accelerations(positions, masses, softening, num_bodies)
            dv_dt = accelerations.reshape(-1)
            
            return np.concatenate([dp_dt, dv_dt])
        
        rtol = 1e-8
        atol = 1e-8
        method = "RK45"
        
        sol = solve_ivp(
            nbodyproblem,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=None,
            dense_output=False
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1].tolist()