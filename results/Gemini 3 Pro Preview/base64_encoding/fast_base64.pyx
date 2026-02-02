import cython
from cython.parallel import prange
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString

cdef uint8_t[64] TABLE
cdef bytes B64_CHARS = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
cdef int i
for i in range(64):
    TABLE[i] = B64_CHARS[i]

cdef uint16_t[4096] T12
cdef int j
cdef uint8_t c1, c2
for j in range(4096):
    c1 = TABLE[j >> 6]
    c2 = TABLE[j & 0x3f]
    # Little endian: c1 at low address (LSB), c2 at high address (MSB)
    T12[j] = (c2 << 8) | c1

@cython.boundscheck(False)
@cython.wraparound(False)
def b64encode(const uint8_t[:] data):
    cdef Py_ssize_t n = data.shape[0]
    if n == 0:
        return b""
        
    cdef Py_ssize_t out_len = 4 * ((n + 2) // 3)
    cdef bytes res = PyBytes_FromStringAndSize(NULL, out_len)
    cdef char* res_ptr = PyBytes_AsString(res)
    cdef const uint8_t* data_ptr = &data[0]
    
    cdef Py_ssize_t limit = n // 3
    cdef Py_ssize_t k, idx_in, idx_out
    cdef uint32_t b1, b2, b3
    cdef uint64_t w1, w2, w3, w4
    
    # Release GIL for parallelism
    with nogil:
        # Unroll loop by 8 (24 bytes input -> 32 bytes output)
        for k in prange(0, limit - (limit % 8), 8, schedule='static'):
            idx_in = k * 3
            idx_out = k * 4
            
            # 1 & 2
            b1 = data_ptr[idx_in]
            b2 = data_ptr[idx_in+1]
            b3 = data_ptr[idx_in+2]
            w1 = T12[(b1 << 4) | (b2 >> 4)] | (<uint64_t>T12[((b2 & 0x0f) << 8) | b3] << 16)
            
            b1 = data_ptr[idx_in+3]
            b2 = data_ptr[idx_in+4]
            b3 = data_ptr[idx_in+5]
            w1 = w1 | (<uint64_t>T12[(b1 << 4) | (b2 >> 4)] << 32) | (<uint64_t>T12[((b2 & 0x0f) << 8) | b3] << 48)
            (<uint64_t*>(res_ptr + idx_out))[0] = w1
            
            # 3 & 4
            b1 = data_ptr[idx_in+6]
            b2 = data_ptr[idx_in+7]
            b3 = data_ptr[idx_in+8]
            w2 = T12[(b1 << 4) | (b2 >> 4)] | (<uint64_t>T12[((b2 & 0x0f) << 8) | b3] << 16)
            
            b1 = data_ptr[idx_in+9]
            b2 = data_ptr[idx_in+10]
            b3 = data_ptr[idx_in+11]
            w2 = w2 | (<uint64_t>T12[(b1 << 4) | (b2 >> 4)] << 32) | (<uint64_t>T12[((b2 & 0x0f) << 8) | b3] << 48)
            (<uint64_t*>(res_ptr + idx_out + 8))[0] = w2

            # 5 & 6
            b1 = data_ptr[idx_in+12]
            b2 = data_ptr[idx_in+13]
            b3 = data_ptr[idx_in+14]
            w3 = T12[(b1 << 4) | (b2 >> 4)] | (<uint64_t>T12[((b2 & 0x0f) << 8) | b3] << 16)
            
            b1 = data_ptr[idx_in+15]
            b2 = data_ptr[idx_in+16]
            b3 = data_ptr[idx_in+17]
            w3 = w3 | (<uint64_t>T12[(b1 << 4) | (b2 >> 4)] << 32) | (<uint64_t>T12[((b2 & 0x0f) << 8) | b3] << 48)
            (<uint64_t*>(res_ptr + idx_out + 16))[0] = w3

            # 7 & 8
            b1 = data_ptr[idx_in+18]
            b2 = data_ptr[idx_in+19]
            b3 = data_ptr[idx_in+20]
            w4 = T12[(b1 << 4) | (b2 >> 4)] | (<uint64_t>T12[((b2 & 0x0f) << 8) | b3] << 16)
            
            b1 = data_ptr[idx_in+21]
            b2 = data_ptr[idx_in+22]
            b3 = data_ptr[idx_in+23]
            w4 = w4 | (<uint64_t>T12[(b1 << 4) | (b2 >> 4)] << 32) | (<uint64_t>T12[((b2 & 0x0f) << 8) | b3] << 48)
            (<uint64_t*>(res_ptr + idx_out + 24))[0] = w4

        # Handle remaining chunks
        for k in range(limit - (limit % 8), limit):
            idx_in = k * 3
            idx_out = k * 4
            b1 = data_ptr[idx_in]
            b2 = data_ptr[idx_in+1]
            b3 = data_ptr[idx_in+2]
            (<uint16_t*>(res_ptr + idx_out))[0] = T12[(b1 << 4) | (b2 >> 4)]
            (<uint16_t*>(res_ptr + idx_out))[1] = T12[((b2 & 0x0f) << 8) | b3]
            b3 = data_ptr[idx_in+17]
            (<uint16_t*>(res_ptr + idx_out + 20))[0] = T12[(b1 << 4) | (b2 >> 4)]
            (<uint16_t*>(res_ptr + idx_out + 20))[1] = T12[((b2 & 0x0f) << 8) | b3]

            # 7
            b1 = data_ptr[idx_in+18]
            b2 = data_ptr[idx_in+19]
            b3 = data_ptr[idx_in+20]
            (<uint16_t*>(res_ptr + idx_out + 24))[0] = T12[(b1 << 4) | (b2 >> 4)]
            (<uint16_t*>(res_ptr + idx_out + 24))[1] = T12[((b2 & 0x0f) << 8) | b3]

            # 8
            b1 = data_ptr[idx_in+21]
            b2 = data_ptr[idx_in+22]
            b3 = data_ptr[idx_in+23]
            (<uint16_t*>(res_ptr + idx_out + 28))[0] = T12[(b1 << 4) | (b2 >> 4)]
            (<uint16_t*>(res_ptr + idx_out + 28))[1] = T12[((b2 & 0x0f) << 8) | b3]

        # Handle remaining chunks
        for k in range(limit - (limit % 8), limit):
            idx_in = k * 3
            idx_out = k * 4
            b1 = data_ptr[idx_in]
            b2 = data_ptr[idx_in+1]
            b3 = data_ptr[idx_in+2]
            (<uint16_t*>(res_ptr + idx_out))[0] = T12[(b1 << 4) | (b2 >> 4)]
            (<uint16_t*>(res_ptr + idx_out))[1] = T12[((b2 & 0x0f) << 8) | b3]
    # Padding (serial)
    cdef Py_ssize_t rem = n % 3
    if rem > 0:
        idx_in = limit * 3
        idx_out = limit * 4
        if rem == 1:
            b1 = data_ptr[idx_in]
            res_ptr[idx_out] = TABLE[b1 >> 2]
            res_ptr[idx_out+1] = TABLE[(b1 & 0x03) << 4]
            res_ptr[idx_out+2] = 61 # '='
            res_ptr[idx_out+3] = 61 # '='
        elif rem == 2:
            b1 = data_ptr[idx_in]
            b2 = data_ptr[idx_in+1]
            res_ptr[idx_out] = TABLE[b1 >> 2]
            res_ptr[idx_out+1] = TABLE[((b1 & 0x03) << 4) | (b2 >> 4)]
            res_ptr[idx_out+2] = TABLE[(b2 & 0x0f) << 2]
            res_ptr[idx_out+3] = 61 # '='
            
    return res