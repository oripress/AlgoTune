# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport exp

cdef inline double alpha_m_func(double V) nogil:
    if V == -40.0:
        return 1.0
    else:
        return 0.1 * (V + 40.0) / (1.0 - exp(-(V + 40.0) / 10.0))

cdef inline double beta_m_func(double V) nogil:
    return 4.0 * exp(-(V + 65.0) / 18.0)

cdef inline double alpha_h_func(double V) nogil:
    return 0.07 * exp(-(V + 65.0) / 20.0)

cdef inline double beta_h_func(double V) nogil:
    return 1.0 / (1.0 + exp(-(V + 35.0) / 10.0))

cdef inline double alpha_n_func(double V) nogil:
    if V == -55.0:
        return 0.1
    else:
        return 0.01 * (V + 55.0) / (1.0 - exp(-(V + 55.0) / 10.0))

cdef inline double beta_n_func(double V) nogil:
    return 0.125 * exp(-(V + 65.0) / 80.0)

def hodgkin_huxley_cython(double t, double[::1] y, double C_m, double g_Na, double g_K, 
                          double g_L, double E_Na, double E_K, double E_L, double I_app):
    cdef double V = y[0]
    cdef double m = y[1]
    cdef double h = y[2]
    cdef double n = y[3]
    
    cdef double alpha_m = alpha_m_func(V)
    cdef double beta_m = beta_m_func(V)
    cdef double alpha_h = alpha_h_func(V)
    cdef double beta_h = beta_h_func(V)
    cdef double alpha_n = alpha_n_func(V)
    cdef double beta_n = beta_n_func(V)
    
    cdef double m3 = m * m * m
    cdef double n4 = n * n * n * n
    cdef double I_Na = g_Na * m3 * h * (V - E_Na)
    cdef double I_K = g_K * n4 * (V - E_K)
    cdef double I_L = g_L * (V - E_L)
    
    cdef double dVdt = (I_app - I_Na - I_K - I_L) / C_m
    cdef double dmdt = alpha_m * (1.0 - m) - beta_m * m
    cdef double dhdt = alpha_h * (1.0 - h) - beta_h * h
    cdef double dndt = alpha_n * (1.0 - n) - beta_n * n
    
    return np.array([dVdt, dmdt, dhdt, dndt], dtype=np.float64)