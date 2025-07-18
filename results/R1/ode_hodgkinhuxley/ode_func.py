import numpy as np
from numba import njit

@njit(cache=True)
def hodgkin_huxley(t, y, params):
    V, m, h, n = y
    C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app = params

    # Precompute common terms
    V40 = V + 40.0
    V55 = V + 55.0
    V65 = V + 65.0
    
    # Alpha_m with singularity handling
    if abs(V40) < 1e-8:
        alpha_m = 1.0
    else:
        alpha_m = 0.1 * V40 / (1.0 - np.exp(-V40 / 10.0))
    
    beta_m = 4.0 * np.exp(-V65 / 18.0)
    
    alpha_h = 0.07 * np.exp(-V65 / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    # Alpha_n with singularity handling
    if abs(V55) < 1e-8:
        alpha_n = 0.1
    else:
        alpha_n = 0.01 * V55 / (1.0 - np.exp(-V55 / 10.0))
    
    beta_n = 0.125 * np.exp(-V65 / 80.0)
    
    # Clip gating variables
    m = np.clip(m, 0.0, 1.0)
    h = np.clip(h, 0.0, 1.0)
    n = np.clip(n, 0.0, 1.0)
    
    # Current calculations
    I_Na = g_Na * (m**3) * h * (V - E_Na)
    I_K = g_K * (n**4) * (V - E_K)
    I_L = g_L * (V - E_L)
    
    # Derivatives
    dVdt = (I_app - I_Na - I_K - I_L) / C_m
    dmdt = alpha_m * (1.0 - m) - beta_m * m
    dhdt = alpha_h * (1.0 - h) - beta_h * h
    dndt = alpha_n * (1.0 - n) - beta_n * n
    
    return np.array([dVdt, dmdt, dhdt, dndt])