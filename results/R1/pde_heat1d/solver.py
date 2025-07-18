import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        # Extract parameters
        y0 = np.array(problem["y0"])
        t0 = problem["t0"]
        t1 = problem["t1"]
        params = problem["params"]
        alpha = params["alpha"]
        dx = params["dx"]
        n = len(y0)
        dx2 = dx * dx
        
        # Define the heat equation function with vectorized padding
        def heat_equation(t, u):
            # Apply boundary conditions (u=0 at boundaries)
            u_padded = np.zeros(n + 2)
            u_padded[1:-1] = u
            
            # Compute second derivative using vectorized operations
            u_xx = u_padded[2:] - 2 * u_padded[1:-1] + u_padded[:-2]
            u_xx = u_xx / dx2
            
            return alpha * u_xx
        
        # Set solver parameters to match reference
        rtol = 1e-6
        atol = 1e-6
        
        # Solve the ODE system
        sol = solve_ivp(
            heat_equation,
            [t0, t1],
            y0,
            method='RK45',
            rtol=rtol,
            atol=atol,
            dense_output=False
        )
        
        # Return the final solution
        return sol.y[:, -1].tolist()