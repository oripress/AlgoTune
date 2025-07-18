# distutils: language = c++
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs

ctypedef fused real_t:
    float
    double

@cython.binding(True)
def solve_cython(real_t t0, real_t t1, real_t[::1] y0, real_t alpha, 
                real_t beta, real_t delta, real_t gamma, real_t dt_factor=0.01):
    cdef real_t t = t0
    cdef real_t[::1] y = y0.copy()
    cdef real_t dt = (t1 - t0) * dt_factor
    
    cdef real_t x, y_pred, dx_dt, dy_dt
    cdef real_t k1_x, k1_y, k2_x, k2_y, k3_x, k3_y, k4_x, k4_y
    
    while t < t1:
        if t + dt > t1:
            dt = t1 - t
        
        # Current state
        x = y[0]
        y_pred = y[1]
        
        # RK4 method
        # k1
        k1_x = x * (alpha - beta * y_pred)
        k1_y = y_pred * (delta * x - gamma)
        
        # k2
        k2_x = (x + 0.5 * dt * k1_x) * (alpha - beta * (y_pred + 0.5 * dt * k1_y))
        k2_y = (y_pred + 0.5 * dt * k1_y) * (delta * (x + 0.5 * dt * k1_x) - gamma)
        
        # k3
        k3_x = (x + 0.5 * dt * k2_x) * (alpha - beta * (y_pred + 0.5 * dt * k2_y))
        k3_y = (y_pred + 0.5 * dt * k2_y) * (delta * (x + 0.5 * dt * k2_x) - gamma)
        
        # k4
        k4_x = (x + dt * k3_x) * (alpha - beta * (y_pred + dt * k3_y))
        k4_y = (y_pred + dt * k3_y) * (delta * (x + dt * k3_x) - gamma)
        
        # Update state
        y[0] += dt * (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6.0
        y[1] += dt * (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6.0
        t += dt
    
    return np.array([y[0], y[1]])