import numpy as np
from scipy.integrate import solve_ivp
from typing import Any
import numba as nb

class Solver:
    def __init__(self):
        # Pre-compile the ODE function with a dummy call
        self._compile_functions()
    
    def _compile_functions(self):
        """Pre-compile Numba functions with dummy data."""
        dummy_y = np.array([0.0, 0.5, 0.5, 0.5])
        dummy_params = (1.0, 120.0, 36.0, 0.3, 50.0, -77.0, -54.4, 10.0)
        _ = hodgkin_huxley_numba(0.0, dummy_y, dummy_params)
    
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Solve the Hodgkin-Huxley model."""
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        
        # Pack parameters into a tuple for Numba
        params = problem["params"]
        params_tuple = (
            params["C_m"],
            params["g_Na"], 
            params["g_K"],
            params["g_L"],
            params["E_Na"],
            params["E_K"],
            params["E_L"],
            params["I_app"]
        )
        
        # Wrapper function to match scipy's expected signature
        def ode_wrapper(t, y):
            return hodgkin_huxley_numba(t, y, params_tuple)
        
        # Use DOP853 (8th order Runge-Kutta) for better speed/accuracy trade-off
        sol = solve_ivp(
            ode_wrapper,
            [t0, t1],
            y0,
            method="DOP853",
            rtol=1e-7,  # Slightly looser tolerance for speed
            atol=1e-9,
            dense_output=False,
            vectorized=False
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1].tolist()

@nb.njit(fastmath=True, cache=True)
def hodgkin_huxley_numba(t, y, params):
    """Numba-compiled Hodgkin-Huxley ODE function."""
    V, m, h, n = y
    C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app = params
    
    # Pre-compute commonly used values
    V_40 = V + 40.0
    V_55 = V + 55.0
    V_65 = V + 65.0
    V_35 = V + 35.0
    
    # Alpha_m with singularity handling
    if abs(V_40) < 1e-7:
        alpha_m = 1.0
    else:
        exp_val = np.exp(-V_40 / 10.0)
        alpha_m = 0.1 * V_40 / (1.0 - exp_val)
    
    beta_m = 4.0 * np.exp(-V_65 / 18.0)
    
    alpha_h = 0.07 * np.exp(-V_65 / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-V_35 / 10.0))
    
    # Alpha_n with singularity handling
    if abs(V_55) < 1e-7:
        alpha_n = 0.1
    else:
        exp_val = np.exp(-V_55 / 10.0)
        alpha_n = 0.01 * V_55 / (1.0 - exp_val)
    
    beta_n = 0.125 * np.exp(-V_65 / 80.0)
    
    # Clip gating variables (ensure physical bounds)
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
    
    result = np.empty(4)
    result[0] = dVdt
    result[1] = dmdt
    result[2] = dhdt
    result[3] = dndt
    
    return result