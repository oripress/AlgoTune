# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False

from cython.parallel import prange
import numpy as np
cimport numpy as np
cimport cython
import sys

# --- C-level LUTs for maximum performance ---
cdef unsigned char[64] B64_CHARS
B64_CHARS[:] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
cdef unsigned char PAD_CHAR = b'='[0]

# Pre-computed 12-bit to uint16 Lookup Table
cdef unsigned short[4096] LUT12_U16

# This must be called from Python to initialize the LUT with correct endianness.
def _init_lut():
    if sys.byteorder == 'little':
        for i in range(4096):
            LUT12_U16[i] = B64_CHARS[(i >> 6) & 0x3F] | (B64_CHARS[i & 0x3F] << 8)
    else: # big-endian
        for i in range(4096):
            LUT12_U16[i] = (B64_CHARS[(i >> 6) & 0x3F] << 8) | B64_CHARS[i & 0x3F]

# The main parallel Cython function using block-based processing.
def b64encode_cython_parallel(np.ndarray[np.uint8_t, ndim=1, mode="c"] data):
    cdef int n = data.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.uint8)

    cdef int out_len = ((n + 2) // 3) * 4
    cdef np.ndarray[np.uint8_t, ndim=1, mode="c"] encoded = np.empty(out_len, dtype=np.uint8)

    # Get raw pointers for high-speed, no-overhead access
    cdef unsigned char* p_data = <unsigned char*>data.data
    cdef unsigned short* p_encoded_u16 = <unsigned short*>encoded.data

    cdef int num_chunks = n // 3
    
    # Define a block size in chunks. A large size ensures threads work on
    # independent memory regions, avoiding cache contention.
    cdef int block_size_chunks = 4096 # 12KB input per block
    cdef int num_blocks = (num_chunks + block_size_chunks - 1) // block_size_chunks

    # The parallel section. The GIL is released via `nogil`.
    # We parallelize over the BLOCKS of data.
    with nogil:
        for block_idx in prange(num_blocks, schedule='static'):
            cdef int start_chunk = block_idx * block_size_chunks
            cdef int end_chunk = start_chunk + block_size_chunks
            if end_chunk > num_chunks:
                end_chunk = num_chunks

            # Fast sequential inner loop for the block, using the uint16 LUT
            cdef int i
            cdef unsigned int chunk
            for i in range(start_chunk, end_chunk):
                chunk = (p_data[i*3] << 16) | (p_data[i*3 + 1] << 8) | p_data[i*3 + 2]
                p_encoded_u16[i*2] = LUT12_U16[chunk >> 12]
                p_encoded_u16[i*2 + 1] = LUT12_U16[chunk & 0xFFF]

    # Remainder (padding) is handled sequentially by the main thread.
    cdef int rem = n % 3
    if rem > 0:
        cdef int out_idx_u8 = num_chunks * 4
        cdef int rem_idx = num_chunks * 3
        cdef unsigned char* p_encoded_u8 = <unsigned char*>encoded.data
        if rem == 1:
            chunk = p_data[rem_idx] << 16
            p_encoded_u8[out_idx_u8]     = B64_CHARS[(chunk >> 18) & 0x3F]
            p_encoded_u8[out_idx_u8 + 1] = B64_CHARS[(chunk >> 12) & 0x3F]
            p_encoded_u8[out_idx_u8 + 2] = PAD_CHAR
            p_encoded_u8[out_idx_u8 + 3] = PAD_CHAR
        else: # rem == 2
            chunk = (p_data[rem_idx] << 16) | (p_data[rem_idx + 1] << 8)
            p_encoded_u8[out_idx_u8]     = B64_CHARS[(chunk >> 18) & 0x3F]
            p_encoded_u8[out_idx_u8 + 1] = B64_CHARS[(chunk >> 12) & 0x3F]
            p_encoded_u8[out_idx_u8 + 2] = B64_CHARS[(chunk >> 6) & 0x3F]
            p_encoded_u8[out_idx_u8 + 3] = PAD_CHAR

    return encoded