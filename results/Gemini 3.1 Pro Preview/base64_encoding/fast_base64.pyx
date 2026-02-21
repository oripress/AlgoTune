# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from libc.stdint cimport uint8_t, uint16_t, uint32_t
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING

cdef const uint8_t* alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
cdef uint16_t[4096] lut
cdef bint lut_initialized = False

cdef void init_lut():
    cdef int i
    cdef uint8_t c1, c2
    for i in range(4096):
        c1 = alphabet[(i >> 6) & 0x3F]
        c2 = alphabet[i & 0x3F]
        lut[i] = (<uint16_t>c2 << 8) | c1

cpdef bytes encode(const uint8_t[:] data):
    global lut_initialized
    if not lut_initialized:
        init_lut()
        lut_initialized = True

    cdef Py_ssize_t n = data.shape[0]
    cdef Py_ssize_t out_len = 4 * ((n + 2) // 3)
    
    cdef bytes out_bytes = PyBytes_FromStringAndSize(NULL, out_len)
    if n == 0:
        return out_bytes
        
    cdef uint8_t* out = <uint8_t*>PyBytes_AS_STRING(out_bytes)
    cdef uint32_t* out32 = <uint32_t*>out
    cdef const uint8_t* in_ptr = &data[0]
    
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef uint32_t b0, b1, b2, combined
    
    while i + 11 < n:
        b0 = in_ptr[i]
        b1 = in_ptr[i+1]
        b2 = in_ptr[i+2]
        combined = (b0 << 16) | (b1 << 8) | b2
        out32[j] = (<uint32_t>lut[combined & 0xFFF] << 16) | lut[combined >> 12]
        
        b0 = in_ptr[i+3]
        b1 = in_ptr[i+4]
        b2 = in_ptr[i+5]
        combined = (b0 << 16) | (b1 << 8) | b2
        out32[j+1] = (<uint32_t>lut[combined & 0xFFF] << 16) | lut[combined >> 12]
        
        b0 = in_ptr[i+6]
        b1 = in_ptr[i+7]
        b2 = in_ptr[i+8]
        combined = (b0 << 16) | (b1 << 8) | b2
        out32[j+2] = (<uint32_t>lut[combined & 0xFFF] << 16) | lut[combined >> 12]
        
        b0 = in_ptr[i+9]
        b1 = in_ptr[i+10]
        b2 = in_ptr[i+11]
        combined = (b0 << 16) | (b1 << 8) | b2
        out32[j+3] = (<uint32_t>lut[combined & 0xFFF] << 16) | lut[combined >> 12]
        
        i += 12
        j += 4

    while i + 2 < n:
        b0 = in_ptr[i]
        b1 = in_ptr[i+1]
        b2 = in_ptr[i+2]
        combined = (b0 << 16) | (b1 << 8) | b2
        out32[j] = (<uint32_t>lut[combined & 0xFFF] << 16) | lut[combined >> 12]
        i += 3
        j += 1
        
    j = j * 4
    if i < n:
        b0 = in_ptr[i]
        out[j] = alphabet[b0 >> 2]
        if i + 1 < n:
            b1 = in_ptr[i+1]
            out[j+1] = alphabet[((b0 & 0x03) << 4) | (b1 >> 4)]
            out[j+2] = alphabet[(b1 & 0x0F) << 2]
            out[j+3] = 61 # '='
        else:
            out[j+1] = alphabet[(b0 & 0x03) << 4]
            out[j+2] = 61 # '='
            out[j+3] = 61 # '='
            
    return out_bytes