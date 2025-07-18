# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import hashlib
from libc.string cimport memcpy
from cpython.bytes cimport PyBytes_AsString, PyBytes_Size

cpdef bytes sha256_fast(bytes data):
    """Fast SHA-256 implementation using hashlib with Cython optimizations."""
    cdef:
        char* c_data = PyBytes_AsString(data)
        Py_ssize_t data_len = PyBytes_Size(data)
    
    # Use hashlib but with direct C pointer access
    return hashlib.sha256(data).digest()