import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def hodgkin_huxley_cython(double t, np.ndarray[np.double_t, ndim=1] y, 
                         double C_m, double g_Na, double g_K, double g_L, 
                         double E_Na, double E_K, double E_L, double I_app):
    cdef double V = y[0]
    cdef double m = y[1]
    cdef double h = y[2]
    cdef double n = y[3]
    
    # Precompute common terms
    cdef double V40 = V + 40.0
    cdef double V55 = V + 55.0
    cdef double V65 = V + 65.0
    
    # Alpha_m with singularity handling
    cdef double alpha_m
    if abs(V40) < 1e-8:
        alpha_m = 1.0
    else:
        alpha_m = 0.1 * V40 / (1.0 - np.exp(-V40 / 10.0))
    
    cdef double beta_m = 4.0 * np.exp(-V65 / 18.0)
    
    cdef double alpha_h = 0.07 * np.exp(-V65 / 20.0)
    cdef double beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    # Alpha_n with singularity handling
    cdef double alpha_n
    if abs(V55) < 1e-8:
        alpha_n = 0.1
    else:
        alpha_n = 0.01 * V55 / (1.0 - np.exp(-V55 / 10.0))
    
    cdef double beta_n = 0.125 * np.exp(-V65 / 80.0)
    
    # Clip gating variables
    if m < 0.0: m = 0.0
    elif m > 1.0: m = 1.0
    if h < 0.0: h = 0.0
    elif h > 1.0: h = 1.0
    if n < 0.0: n = 0.0
    elif n > 1.0: n = 1.0
    
    # Current calculations (optimized)
    cdef double m3 = m*m*m
    cdef double n4 = n*n*n*n
    cdef double I_Na = g_Na * m3 * h * (V - E_Na)
    cdef double I_K = g_K * n4 * (V - E_K)
    cdef double I_L = g_L * (V - E_L)
    
    # Derivatives
    cdef double dVdt = (I_app - I_Na - I_K - I_L) / C_m
    cdef double dmdt = alpha_m * (1.0 - m) - beta_m * m
    cdef double dhdt = alpha_h * (1.0 - h) - beta_h * h
    cdef double dndt = alpha_n * (1.0 - n) - beta_n * n
    
    return np.array([dVdt, dmdt, dhdt, dndt])