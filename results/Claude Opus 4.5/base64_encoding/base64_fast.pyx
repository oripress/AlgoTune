# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
cimport cython
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING
from cython.parallel import prange

cdef const unsigned char* ALPHABET = b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bytes base64_encode_fast(bytes data):
    cdef:
        Py_ssize_t n = len(data)
        Py_ssize_t remainder = n % 3
        Py_ssize_t pad_len = (3 - remainder) % 3 if remainder else 0
        Py_ssize_t n_groups = (n + pad_len) // 3
        Py_ssize_t out_len = n_groups * 4
        Py_ssize_t complete_groups = n // 3
        Py_ssize_t i, idx, out_idx
        unsigned char b0, b1, b2
        const unsigned char* src = <const unsigned char*>data
        unsigned char* dst
        bytes result
    
    result = PyBytes_FromStringAndSize(NULL, out_len)
    dst = <unsigned char*>PyBytes_AS_STRING(result)
    
    # Process complete groups of 3 bytes in parallel
    for i in prange(complete_groups, nogil=True, schedule='static'):
        idx = i * 3
        out_idx = i * 4
        b0 = src[idx]
        b1 = src[idx + 1]
        b2 = src[idx + 2]
        
        dst[out_idx] = ALPHABET[b0 >> 2]
        dst[out_idx + 1] = ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)]
        dst[out_idx + 2] = ALPHABET[((b1 & 0x0F) << 2) | (b2 >> 6)]
        dst[out_idx + 3] = ALPHABET[b2 & 0x3F]
    
    # Handle remainder
    if remainder == 1:
        idx = complete_groups * 3
        out_idx = complete_groups * 4
        b0 = src[idx]
        dst[out_idx] = ALPHABET[b0 >> 2]
        dst[out_idx + 1] = ALPHABET[(b0 & 0x03) << 4]
        dst[out_idx + 2] = 61  # '='
        dst[out_idx + 3] = 61  # '='
    elif remainder == 2:
        idx = complete_groups * 3
        out_idx = complete_groups * 4
        b0 = src[idx]
        b1 = src[idx + 1]
        dst[out_idx] = ALPHABET[b0 >> 2]
        dst[out_idx + 1] = ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)]
        dst[out_idx + 2] = ALPHABET[(b1 & 0x0F) << 2]
        dst[out_idx + 3] = 61  # '='
    
    return result