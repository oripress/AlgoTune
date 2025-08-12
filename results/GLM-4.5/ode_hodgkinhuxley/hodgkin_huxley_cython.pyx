# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def hodgkin_huxley_rhs(double t, double[::1] y, double C_m, double g_Na, double g_K, 
                      double g_L, double E_Na, double E_K, double E_L, double I_app):
    # Unpack state variables
    cdef double V = y[0]
    cdef double m = y[1]
    cdef double h = y[2]
    cdef double n = y[3]
    
    # Rate constants
    cdef double alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n
    
    # Calculate alpha and beta rate constants
    # Handle singularities in rate functions (when denominator approaches 0)
    if V == -40.0:
        alpha_m = 1.0  # L'Hôpital's rule limit at V = -40.0
    else:
        alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

    cdef double V_plus_65 = V + 65.0
    beta_m = 4.0 * np.exp(-V_plus_65 / 18.0)

    alpha_h = 0.07 * np.exp(-V_plus_65 / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    # Handle singularity in alpha_n at V = -55.0
    if V == -55.0:
        alpha_n = 0.1  # L'Hôpital's rule limit at V = -55.0
    else:
        alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    beta_n = 0.125 * np.exp(-V_plus_65 / 80.0)

    # Ensure gating variables stay in [0, 1]
    m = max(0.0, min(1.0, m))
    h = max(0.0, min(1.0, h))
    n = max(0.0, min(1.0, n))

    # Calculate ionic currents with optimized operations
    cdef double V_minus_E_Na = V - E_Na
    cdef double V_minus_E_K = V - E_K
    cdef double V_minus_E_L = V - E_L
    
    # Use power operator for better performance
    cdef double m3 = m * m * m
    cdef double n4 = n * n * n * n
    
    cdef double I_Na = g_Na * m3 * h * V_minus_E_Na
    cdef double I_K = g_K * n4 * V_minus_E_K
    cdef double I_L = g_L * V_minus_E_L

    # Differential equations
    cdef double dVdt = (I_app - I_Na - I_K - I_L) / C_m
    cdef double dmdt = alpha_m * (1.0 - m) - beta_m * m
    cdef double dhdt = alpha_h * (1.0 - h) - beta_h * h
    cdef double dndt = alpha_n * (1.0 - n) - beta_n * n

    return np.array([dVdt, dmdt, dhdt, dndt])