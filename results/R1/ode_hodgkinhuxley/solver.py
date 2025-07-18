import numpy as np
from numba import njit

@njit
def hodgkin_huxley_numba(t, y, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app):
    V, m, h, n = y
    
    # Calculate alpha and beta rate constants
    # Handle singularities in rate functions
    if V == -40.0:
        alpha_m = 1.0
    else:
        alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    
    beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
    alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    if V == -55.0:
        alpha_n = 0.1
    else:
        alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    
    beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    # Ensure gating variables stay in [0, 1] using manual clamping
    m = min(1.0, max(0.0, m))
    h = min(1.0, max(0.0, h))
    n = min(1.0, max(0.0, n))
    
    # Calculate ionic currents
    I_Na = g_Na * (m ** 3) * h * (V - E_Na)
    I_K = g_K * (n ** 4) * (V - E_K)
    I_L = g_L * (V - E_L)
    
    # Differential equations
    dVdt = (I_app - I_Na - I_K - I_L) / C_m
    dmdt = alpha_m * (1.0 - m) - beta_m * m
    dhdt = alpha_h * (1.0 - h) - beta_h * h
    dndt = alpha_n * (1.0 - n) - beta_n * n
    
    return np.array([dVdt, dmdt, dhdt, dndt])

@njit
def rk4_step(f, t, y, h, args):
    """Single RK4 step."""
    k1 = f(t, y, *args)
    k2 = f(t + h/2, y + h/2 * k1, *args)
    k3 = f(t + h/2, y + h/2 * k2, *args)
    k4 = f(t + h, y + h * k3, *args)
    y_next = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    return y_next

@njit
def _integrate(t0: float, t1: float, y0: np.ndarray, 
             C_m: float, g_Na: float, g_K: float, g_L: float, 
             E_Na: float, E_K: float, E_L: float, I_app: float) -> np.ndarray:
    """Adaptive RK4 integrator with simple step size control."""
    t = t0
    y = y0.copy()
    dt = 0.01  # Initial step size
    args = (C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    min_dt = 1e-4
    max_dt = 0.1
    
    while t < t1:
        if t + dt > t1:
            dt = t1 - t
            
        # Take two half steps
        y_half = rk4_step(hodgkin_huxley_numba, t, y, dt/2, args)
        y_full = rk4_step(hodgkin_huxley_numba, t + dt/2, y_half, dt/2, args)
        
        # Take one full step
        y_single = rk4_step(hodgkin_huxley_numba, t, y, dt, args)
        
        # Error estimate
        error = np.max(np.abs(y_full - y_single))
        
        # Adjust step size based on error
        if error < 1e-6:
            t += dt
            y = y_full
            # Increase step size but cap at max_dt
            dt = min(max_dt, dt * 1.5)
        else:
            # Reduce step size but don't go below min_dt
            dt = max(min_dt, dt / 2)
                
    return y

def integrate(t0: float, t1: float, y0: np.ndarray, 
             C_m: float, g_Na: float, g_K: float, g_L: float, 
             E_Na: float, E_K: float, E_L: float, I_app: float) -> np.ndarray:
    """Wrapper for the Numba-compiled integrator."""
    return _integrate(t0, t1, y0, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)

class Solver:
    def solve(self, problem, **kwargs):
        # Extract parameters
        t0 = float(problem['t0'])
        t1 = float(problem['t1'])
        y0 = np.array(problem['y0'], dtype=np.float64)
        p = problem['params']
        
        C_m = float(p['C_m'])
        g_Na = float(p['g_Na'])
        g_K = float(p['g_K'])
        g_L = float(p['g_L'])
        E_Na = float(p['E_Na'])
        E_K = float(p['E_K'])
        E_L = float(p['E_L'])
        I_app = float(p['I_app'])
            
        # Call the integrator
        result = integrate(t0, t1, y0, 
                          C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
        
        return result.tolist()