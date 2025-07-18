# distutils: extra_compile_args = -O3 -march=native

from cpython cimport PyBytes_FromStringAndSize
from libc.stdlib cimport malloc, free
cimport cython

# Precompute lookup table for all 3-byte combinations
cdef unsigned char* lookup_table = <unsigned char*> malloc(256 * 256 * 256 * 4 * sizeof(unsigned char))
cdef char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

# Initialize lookup table
cdef int i, j, k, idx
cdef unsigned char a, b, c
for i in range(256):
    for j in range(256):
        for k in range(256):
            idx = (i << 16) | (j << 8) | k
            lookup_table[idx*4] = alphabet[i >> 2]
            lookup_table[idx*4+1] = alphabet[((i & 0x03) << 4) | (j >> 4)]
            lookup_table[idx*4+2] = alphabet[((j & 0x0F) << 2) | (k >> 6)]
            lookup_table[idx*4+3] = alphabet[k & 0x3F]

@cython.boundscheck(False)
@cython.wraparound(False)
def base64_encode(bytes data):
    cdef:
        const unsigned char* in_ptr = data
        int in_len = len(data)
        int out_len = (in_len + 2) // 3 * 4
        unsigned char* out_buf = <unsigned char*> malloc(out_len * sizeof(unsigned char))
        int chunks = in_len // 3
        int remainder = in_len % 3
        int i, j = 0
        unsigned int triplet
    
    if out_buf is NULL:
        raise MemoryError()
    
    # Process full chunks using lookup table
    for i in range(chunks):
        triplet = (in_ptr[0] << 16) | (in_ptr[1] << 8) | in_ptr[2]
        out_buf[j] = lookup_table[triplet*4]
        out_buf[j+1] = lookup_table[triplet*4+1]
        out_buf[j+2] = lookup_table[triplet*4+2]
        out_buf[j+3] = lookup_table[triplet*4+3]
        in_ptr += 3
        j += 4

    # Process remainder
    if remainder == 1:
        out_buf[j] = alphabet[in_ptr[0] >> 2]
        out_buf[j+1] = alphabet[(in_ptr[0] & 0x03) << 4]
        out_buf[j+2] = 61  # '='
        out_buf[j+3] = 61  # '='
    elif remainder == 2:
        out_buf[j] = alphabet[in_ptr[0] >> 2]
        out_buf[j+1] = alphabet[((in_ptr[0] & 0x03) << 4) | (in_ptr[1] >> 4)]
        out_buf[j+2] = alphabet[(in_ptr[1] & 0x0F) << 2]
        out_buf[j+3] = 61  # '='
    
    cdef bytes result = PyBytes_FromStringAndSize(<char*>out_buf, out_len)
    free(out_buf)
    return result