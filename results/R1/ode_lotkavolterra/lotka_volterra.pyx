import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def lotka_volterra(double t, double[::1] y, double alpha, double beta, double delta, double gamma):
    cdef double x = y[0]
    cdef double y_pred = y[1]
    cdef double dx_dt = alpha * x - beta * x * y_pred
    cdef double dy_dt = delta * x * y_pred - gamma * y_pred
    return np.array([dx_dt, dy_dt])
cimport cython
import numpy as np
cimport numpy as cnp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lotka_volterra(double t, double[:] y, double alpha, double beta, double delta, double gamma):
    cdef double x = y[0]
    cdef double y_val = y[1]
    cdef double xy = x * y_val
    cdef double dx_dt = alpha * x - beta * xy
    cdef double dy_dt = delta * xy - gamma * y_val
    return np.array([dx_dt, dy_dt])
cimport cython
import numpy as np
cimport numpy as cnp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lotka_volterra(double t, double[:] y, double alpha, double beta, double delta, double gamma):
    cdef double x = y[0]
    cdef double y_val = y[1]
    cdef double xy = x * y_val
    cdef double dx_dt = alpha * x - beta * xy
    cdef double dy_dt = delta * xy - gamma * y_val
    return np.array([dx_dt, dy_dt])