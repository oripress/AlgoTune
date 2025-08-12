import numpy as np
from scipy.integrate import solve_ivp
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        # Extract parameters
        y0 = np.array(problem["y0"])
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        F = float(problem["F"])
        
        # Define the Lorenz 96 system dynamics
        # Define the Lorenz 96 system dynamics
        def lorenz96(t, x):
            # Using the standard Lorenz 96 equations:
            # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
            # with cyclic boundary conditions
            
            # Precompute indices for cyclic boundaries
            N = len(x)
            if not hasattr(lorenz96, 'indices'):
                # Create indices only once
                indices = {}
                indices['i_plus_1'] = np.roll(np.arange(N), -1)
                indices['i_minus_1'] = np.roll(np.arange(N), 1)
                indices['i_minus_2'] = np.roll(np.arange(N), 2)
                lorenz96.indices = indices
            
            # Use precomputed indices
            ip1 = lorenz96.indices['i_plus_1']
            im1 = lorenz96.indices['i_minus_1']
            im2 = lorenz96.indices['i_minus_2']
            
            # Compute (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
            dxdt = (x[ip1] - x[im2]) * x[im1] - x + F
            return dxdt

        # Solve the IVP
        sol = solve_ivp(
            fun=lorenz96,
            t_span=[t0, t1],
            y0=y0,
            method='RK45',
            rtol=1e-8,
            atol=1e-8,
            t_eval=[t1],
        )

        if not sol.success:
            raise RuntimeError("Solver failed")

        # Return the final state
        return sol.y[:, -1].tolist()