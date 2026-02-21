# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as np
from libc.math cimport fabs

np.import_array()

def solve_cython(np.ndarray u_obj, np.ndarray v_obj):
    cdef double ans0 = 0.0
    cdef double ans1 = 0.0
    cdef double ans2 = 0.0
    cdef double ans3 = 0.0
    cdef double diff = 0.0
    cdef double d0, d1, d2, d3, d4, d5, d6, d7
    cdef double d01, d23, d45, d67
    cdef double d0123, d4567
    cdef double diff0, diff1, diff2, diff3, diff4, diff5, diff6, diff7
    cdef Py_ssize_t i
    cdef Py_ssize_t n = np.PyArray_SIZE(u_obj)
    cdef double* u = <double*>np.PyArray_DATA(u_obj)
    cdef double* v = <double*>np.PyArray_DATA(v_obj)
    cdef Py_ssize_t n_unroll = n - (n % 8)
    
    for i in range(0, n_unroll, 8):
        d0 = u[i] - v[i]
        d1 = u[i+1] - v[i+1]
        d2 = u[i+2] - v[i+2]
        d3 = u[i+3] - v[i+3]
        d4 = u[i+4] - v[i+4]
        d5 = u[i+5] - v[i+5]
        d6 = u[i+6] - v[i+6]
        d7 = u[i+7] - v[i+7]
        
        d01 = d0 + d1
        d23 = d2 + d3
        d45 = d4 + d5
        d67 = d6 + d7
        
        d0123 = d01 + d23
        d4567 = d45 + d67
        
        diff0 = diff + d0
        diff1 = diff + d01
        diff2 = diff1 + d2
        diff3 = diff + d0123
        diff4 = diff3 + d4
        diff5 = diff3 + d45
        diff6 = diff5 + d6
        diff7 = diff3 + d4567
        
        ans0 += fabs(diff0)
        ans1 += fabs(diff1)
        ans2 += fabs(diff2)
        ans3 += fabs(diff3)
        ans0 += fabs(diff4)
        ans1 += fabs(diff5)
        ans2 += fabs(diff6)
        ans3 += fabs(diff7)
        
        diff = diff7
        
    for i in range(n_unroll, n):
        diff += u[i] - v[i]
        ans0 += fabs(diff)
        
    return ans0 + ans1 + ans2 + ans3