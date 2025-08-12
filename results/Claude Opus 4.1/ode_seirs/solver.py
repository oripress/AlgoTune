import numpy as np
from scipy.integrate import solve_ivp
import numba as nb
from typing import Any

@nb.njit
def seirs_numba(t, y, beta, sigma, gamma, omega):
    """Numba-compiled SEIRS model equations"""
    S = y[0]
    E = y[1]
    I = y[2]
    R = y[3]
    
    dSdt = -beta * S * I + omega * R
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I - omega * R
    
    return np.array([dSdt, dEdt, dIdt, dRdt])

class Solver:
    def __init__(self):
        # Warm up numba compilation
        dummy_y = np.array([0.9, 0.05, 0.03, 0.02], dtype=np.float64)
        _ = seirs_numba(0.0, dummy_y, 0.5, 0.2, 0.1, 0.002)
    
    def solve(self, problem, **kwargs) -> Any:
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = problem["t0"]
        t1 = problem["t1"]
        params = problem["params"]
        
        beta = params["beta"]
        sigma = params["sigma"]
        gamma = params["gamma"]
        omega = params["omega"]
        
        # Create wrapper function for solve_ivp
        def seirs_wrapper(t, y):
            return seirs_numba(t, y, beta, sigma, gamma, omega)
        
        # Use DOP853 (8th order Runge-Kutta) which is often faster for smooth problems
        # with looser tolerances that still maintain accuracy
        rtol = 1e-7
        atol = 1e-9
        
        # Calculate a good initial step size based on problem parameters
        # Smaller omega means slower dynamics
        dt0 = min(1.0 / max(beta, sigma, gamma), 10.0)
        
        # Solve the ODE system
        sol = solve_ivp(
            seirs_wrapper,
            [t0, t1],
            y0,
            method='DOP853',  # High-order method, efficient for smooth problems
            rtol=rtol,
            atol=atol,
            dense_output=False,  # Don't compute dense output to save time
            first_step=dt0,  # Provide initial step size hint
            max_step=np.inf  # Let the solver choose max step
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")