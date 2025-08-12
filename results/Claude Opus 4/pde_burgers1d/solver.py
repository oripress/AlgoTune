import numpy as np
from scipy.integrate import solve_ivp
import numba

class Solver:
    def __init__(self):
        # Pre-compile the numba function
        self._burgers_rhs = self._create_burgers_rhs()
    
    @staticmethod
    @numba.jit(nopython=True)
    def _burgers_equation_numba(u, nu, dx):
        """Compute the right-hand side of Burgers' equation using numba."""
        n = len(u)
        du_dt = np.zeros(n)
        
        # Handle boundaries (u=0 at both ends)
        # Interior points
        for i in range(n):
            # Get values with boundary conditions
            u_left = u[i-1] if i > 0 else 0.0
            u_center = u[i]
            u_right = u[i+1] if i < n-1 else 0.0
            
            # Diffusion term: nu * d²u/dx²
            diffusion = nu * (u_right - 2*u_center + u_left) / (dx*dx)
            
            # Advection term with upwind scheme: u * du/dx
            if u_center >= 0:
                # Use backward difference
                du_dx = (u_center - u_left) / dx
            else:
                # Use forward difference
                du_dx = (u_right - u_center) / dx
            
            advection = u_center * du_dx
            
            # Combine terms
            du_dt[i] = -advection + diffusion
        
        return du_dt
    
    def _create_burgers_rhs(self):
        """Create a closure for the RHS function."""
        burgers_numba = self._burgers_equation_numba
        
        def burgers_rhs(t, u, nu, dx):
            return burgers_numba(u, nu, dx)
        
        return burgers_rhs
    
    def solve(self, problem, **kwargs):
        """Solve the Burgers' equation."""
        y0 = np.array(problem["y0"])
        t0 = problem["t0"]
        t1 = problem["t1"]
        nu = problem["params"]["nu"]
        dx = problem["params"]["dx"]
        
        # Create RHS function with parameters
        def rhs(t, u):
            return self._burgers_rhs(t, u, nu, dx)
        
        # Solve using RK45
        sol = solve_ivp(
            rhs,
            [t0, t1],
            y0,
            method='RK45',
            rtol=1e-6,
            atol=1e-6
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1].tolist()