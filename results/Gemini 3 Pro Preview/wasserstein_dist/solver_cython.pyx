import numpy as np
cimport numpy as np
from libc.math cimport fabs
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solve_cython(object u, object v):
    # Check if inputs are lists
    if isinstance(u, list) and isinstance(v, list):
        return solve_cython_list(u, v)
    
    # Fallback to numpy conversion if not lists (e.g. already numpy arrays)
    cdef double[:] u_view = np.ascontiguousarray(u, dtype=np.float64)
    cdef double[:] v_view = np.ascontiguousarray(v, dtype=np.float64)
    return solve_cython_numpy(u_view, v_view)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double solve_cython_list(list u, list v):
    cdef Py_ssize_t n = len(u)
    if n == 0:
        return 0.0
        
    cdef Py_ssize_t i
    cdef double sum_u = 0.0
    cdef double sum_v = 0.0
    cdef double val_u, val_v
    
    # Pass 1: Calculate sums
    for i in range(n):
        val_u = u[i]
        val_v = v[i]
        sum_u += val_u
        sum_v += val_v
        
    if sum_u == 0.0 or sum_v == 0.0:
        return float(n)
        
    cdef double cum_u = 0.0
    cdef double cum_v = 0.0
    cdef double dist = 0.0
    cdef double inv_prod = 1.0 / (sum_u * sum_v)
    
    # Pass 2: Calculate distance
    for i in range(n - 1):
        val_u = u[i]
        val_v = v[i]
        cum_u += val_u
        cum_v += val_v
        dist += fabs(cum_u * sum_v - cum_v * sum_u)
        
    return dist * inv_prod

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double solve_cython_numpy(const double[:] u, const double[:] v) nogil:
    cdef Py_ssize_t n = u.shape[0]
    if n == 0:
        return 0.0
    cdef const double* u_ptr = &u[0]
    cdef const double* v_ptr = &v[0]
    
    cdef Py_ssize_t i
    cdef double sum_u = 0.0
    cdef double sum_v = 0.0
    
    for i in range(n):
        sum_u += u_ptr[i]
        sum_v += v_ptr[i]
        
    if sum_u == 0.0 or sum_v == 0.0:
        return <double>n
        
    cdef double cum_u = 0.0
    cdef double cum_v = 0.0
    cdef double dist = 0.0
    cdef double inv_prod = 1.0 / (sum_u * sum_v)
    
    for i in range(n - 1):
        cum_u += u_ptr[i]
        cum_v += v_ptr[i]
        dist += fabs(cum_u * sum_v - cum_v * sum_u)
        
    return dist * inv_prod