# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
from libc.math cimport sqrt, fabs, pow

cdef inline double max_d(double a, double b) nogil:
    return a if a > b else b

cdef inline double min_d(double a, double b) nogil:
    return a if a < b else b

cdef inline double abs_d(double a) nogil:
    return a if a >= 0 else -a

cpdef list rk45_solve_cython(double t0, double t1, double y0_0, double y0_1, double alpha, double beta, double delta, double gamma, double rtol, double atol):
    cdef double t = t0
    cdef double y_0 = y0_0
    cdef double y_1 = y0_1
    
    cdef double f0_0 = alpha * y_0 - beta * y_0 * y_1
    cdef double f0_1 = delta * y_0 * y_1 - gamma * y_1
    
    cdef double scale_0 = atol + abs_d(y_0) * rtol
    cdef double scale_1 = atol + abs_d(y_1) * rtol
    cdef double d0 = sqrt(((y_0/scale_0)*(y_0/scale_0) + (y_1/scale_1)*(y_1/scale_1)) / 2.0)
    cdef double d1 = sqrt(((f0_0/scale_0)*(f0_0/scale_0) + (f0_1/scale_1)*(f0_1/scale_1)) / 2.0)
    
    cdef double h0, h1, h_abs, h
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
        
    cdef double y1_0 = y_0 + h0 * f0_0
    cdef double y1_1 = y_1 + h0 * f0_1
    
    cdef double f1_0 = alpha * y1_0 - beta * y1_0 * y1_1
    cdef double f1_1 = delta * y1_0 * y1_1 - gamma * y1_1
    
    scale_0 = atol + max_d(abs_d(y_0), abs_d(y1_0)) * rtol
    scale_1 = atol + max_d(abs_d(y_1), abs_d(y1_1)) * rtol
    
    cdef double d2 = sqrt((((f1_0 - f0_0)/h0/scale_0)*((f1_0 - f0_0)/h0/scale_0) + ((f1_1 - f0_1)/h0/scale_1)*((f1_1 - f0_1)/h0/scale_1)) / 2.0)
    
    cdef double max_d1_d2 = max_d(d1, d2)
    if max_d1_d2 <= 1e-15:
        h1 = max_d(1e-6, h0 * 1e-3)
    else:
        h1 = pow(0.01 / max_d1_d2, 0.2)
        
    h_abs = min_d(100.0 * h0, h1)
    h_abs = max_d(h_abs, 1e-15)
    
    cdef double direction = 1.0 if t1 > t0 else -1.0
    
    cdef double max_factor = 10.0
    cdef double min_factor = 0.2
    cdef double safety = 0.9
    cdef double error_exponent = -0.2
    
    cdef bint step_accepted, step_rejected
    cdef double K0_0, K0_1, K1_0, K1_1, K2_0, K2_1, K3_0, K3_1, K4_0, K4_1, K5_0, K5_1, K6_0, K6_1
    cdef double y_temp_0, y_temp_1, y_new_0, y_new_1, err_0, err_1, error_norm, factor
    
    while t < t1:
        if t + h_abs * direction >= t1:
            h = t1 - t
            h_abs = abs_d(h)
        else:
            h = h_abs * direction
            
        step_accepted = False
        step_rejected = False
        
        while not step_accepted:
            K0_0 = f0_0
            K0_1 = f0_1
            
            y_temp_0 = y_0 + h * (0.2 * K0_0)
            y_temp_1 = y_1 + h * (0.2 * K0_1)
            K1_0 = alpha * y_temp_0 - beta * y_temp_0 * y_temp_1
            K1_1 = delta * y_temp_0 * y_temp_1 - gamma * y_temp_1
            
            y_temp_0 = y_0 + h * (0.075 * K0_0 + 0.225 * K1_0)
            y_temp_1 = y_1 + h * (0.075 * K0_1 + 0.225 * K1_1)
            K2_0 = alpha * y_temp_0 - beta * y_temp_0 * y_temp_1
            K2_1 = delta * y_temp_0 * y_temp_1 - gamma * y_temp_1
            
            y_temp_0 = y_0 + h * (44.0/45.0 * K0_0 - 56.0/15.0 * K1_0 + 32.0/9.0 * K2_0)
            y_temp_1 = y_1 + h * (44.0/45.0 * K0_1 - 56.0/15.0 * K1_1 + 32.0/9.0 * K2_1)
            K3_0 = alpha * y_temp_0 - beta * y_temp_0 * y_temp_1
            K3_1 = delta * y_temp_0 * y_temp_1 - gamma * y_temp_1
            
            y_temp_0 = y_0 + h * (19372.0/6561.0 * K0_0 - 25360.0/2187.0 * K1_0 + 64448.0/6561.0 * K2_0 - 212.0/729.0 * K3_0)
            y_temp_1 = y_1 + h * (19372.0/6561.0 * K0_1 - 25360.0/2187.0 * K1_1 + 64448.0/6561.0 * K2_1 - 212.0/729.0 * K3_1)
            K4_0 = alpha * y_temp_0 - beta * y_temp_0 * y_temp_1
            K4_1 = delta * y_temp_0 * y_temp_1 - gamma * y_temp_1
            
            y_temp_0 = y_0 + h * (9017.0/3168.0 * K0_0 - 355.0/33.0 * K1_0 + 46732.0/5247.0 * K2_0 + 49.0/176.0 * K3_0 - 5103.0/18656.0 * K4_0)
            y_temp_1 = y_1 + h * (9017.0/3168.0 * K0_1 - 355.0/33.0 * K1_1 + 46732.0/5247.0 * K2_1 + 49.0/176.0 * K3_1 - 5103.0/18656.0 * K4_1)
            K5_0 = alpha * y_temp_0 - beta * y_temp_0 * y_temp_1
            K5_1 = delta * y_temp_0 * y_temp_1 - gamma * y_temp_1
            
            y_new_0 = y_0 + h * (35.0/384.0 * K0_0 + 500.0/1113.0 * K2_0 + 125.0/192.0 * K3_0 - 2187.0/6784.0 * K4_0 + 11.0/84.0 * K5_0)
            y_new_1 = y_1 + h * (35.0/384.0 * K0_1 + 500.0/1113.0 * K2_1 + 125.0/192.0 * K3_1 - 2187.0/6784.0 * K4_1 + 11.0/84.0 * K5_1)
            
            K6_0 = alpha * y_new_0 - beta * y_new_0 * y_new_1
            K6_1 = delta * y_new_0 * y_new_1 - gamma * y_new_1
            
            err_0 = h * (-71.0/57600.0 * K0_0 + 71.0/16695.0 * K2_0 - 71.0/1920.0 * K3_0 + 17253.0/339200.0 * K4_0 - 22.0/525.0 * K5_0 + 1.0/40.0 * K6_0)
            err_1 = h * (-71.0/57600.0 * K0_1 + 71.0/16695.0 * K2_1 - 71.0/1920.0 * K3_1 + 17253.0/339200.0 * K4_1 - 22.0/525.0 * K5_1 + 1.0/40.0 * K6_1)
            
            scale_0 = atol + max_d(abs_d(y_0), abs_d(y_new_0)) * rtol
            scale_1 = atol + max_d(abs_d(y_1), abs_d(y_new_1)) * rtol
            
            error_norm = sqrt(((err_0 / scale_0)*(err_0 / scale_0) + (err_1 / scale_1)*(err_1 / scale_1)) / 2.0)
            
            if error_norm < 1.0:
                if error_norm == 0.0:
                    factor = max_factor
                else:
                    factor = min_d(max_factor, safety * pow(error_norm, error_exponent))
                    
                if step_rejected:
                    factor = min_d(1.0, factor)
                    
                h_abs = abs_d(h) * factor
                step_accepted = True
            else:
                factor = max_d(min_factor, safety * pow(error_norm, error_exponent))
                h_abs = abs_d(h) * factor
                h *= factor
                step_rejected = True
                
        t += h
        y_0 = y_new_0
        y_1 = y_new_1
        f0_0 = K6_0
        f0_1 = K6_1
        
    return [y_0, y_1]