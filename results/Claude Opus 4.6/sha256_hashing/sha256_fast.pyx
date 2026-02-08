from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.dict cimport PyDict_GetItem, PyDict_New, PyDict_SetItem

cdef extern from "openssl/sha.h":
    unsigned char *SHA256(const unsigned char *d, size_t n, unsigned char *md)

def solve_sha256(dict problem):
    cdef bytes plaintext = <bytes>problem["plaintext"]
    cdef unsigned char digest[32]
    cdef Py_ssize_t length = len(plaintext)
    SHA256(<const unsigned char *><char *>plaintext, <size_t>length, digest)
    cdef bytes result = PyBytes_FromStringAndSize(<char *>digest, 32)
    return {"digest": result}