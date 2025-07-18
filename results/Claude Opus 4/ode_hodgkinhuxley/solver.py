import numpy as np
from scipy.integrate import solve_ivp
from typing import Any
import numba

@numba.njit
def hodgkin_huxley_jit(t, y, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app):
    # Unpack state variables
    V, m, h, n = y[0], y[1], y[2], y[3]
    
    # Calculate alpha and beta rate constants
    # Handle singularities
    if abs(V + 40.0) < 1e-10:
        alpha_m = 1.0
    else:
        alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    
    beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
    
    alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    if abs(V + 55.0) < 1e-10:
        alpha_n = 0.1
    else:
        alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    
    beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    # Ensure gating variables stay in [0, 1]
    m = min(max(m, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    n = min(max(n, 0.0), 1.0)
    
    # Calculate ionic currents
    m_cubed = m * m * m
    n_fourth = n * n * n * n
    I_Na = g_Na * m_cubed * h * (V - E_Na)
    I_K = g_K * n_fourth * (V - E_K)
    I_L = g_L * (V - E_L)
    
    # Differential equations
    dVdt = (I_app - I_Na - I_K - I_L) / C_m
    dmdt = alpha_m * (1.0 - m) - beta_m * m
    dhdt = alpha_h * (1.0 - h) - beta_h * h
    dndt = alpha_n * (1.0 - n) - beta_n * n
    
    return np.array([dVdt, dmdt, dhdt, dndt])

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Solve the Hodgkin-Huxley neuron model."""
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        
        # Extract parameters
        C_m = params["C_m"]
        g_Na = params["g_Na"]
        g_K = params["g_K"]
        g_L = params["g_L"]
        E_Na = params["E_Na"]
        E_K = params["E_K"]
        E_L = params["E_L"]
        I_app = params["I_app"]
        
        def hodgkin_huxley(t, y):
            return hodgkin_huxley_jit(t, y, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
        
        # Solve ODE with optimized settings
        sol = solve_ivp(
            hodgkin_huxley,
            [t0, t1],
            y0,
            method="DOP853",  # Higher order method
            rtol=1e-7,  # Slightly relaxed tolerance
            atol=1e-8
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")