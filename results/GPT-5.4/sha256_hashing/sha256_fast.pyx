# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize, PyBytes_GET_SIZE
from cpython.dict cimport PyDict_GetItemString
from libc.stddef cimport size_t

cdef extern from "openssl/sha.h":
    unsigned char *SHA256(
        const unsigned char *d,
        size_t n,
        unsigned char *md,
    )

cdef class Solver:
    cdef bytes _out
    cdef dict _result
    cdef unsigned char* _out_ptr

    def __cinit__(self):
        self._out = PyBytes_FromStringAndSize(NULL, 32)
        self._out_ptr = <unsigned char *>PyBytes_AS_STRING(self._out)
        self._result = {"digest": self._out}

    def solve(self, dict problem):
        cdef bytes data = <bytes>PyDict_GetItemString(problem, "plaintext")
        SHA256(
            <const unsigned char *>PyBytes_AS_STRING(data),
            <size_t>PyBytes_GET_SIZE(data),
            self._out_ptr,
        )
        return self._result