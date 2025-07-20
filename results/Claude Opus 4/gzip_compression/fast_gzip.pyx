# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cpython.bytes cimport PyBytes_FromStringAndSize
import struct
import zlib

cdef extern from "zlib.h":
    ctypedef struct z_stream:
        unsigned char* next_in
        unsigned int avail_in
        unsigned long total_in
        unsigned char* next_out
        unsigned int avail_out
        unsigned long total_out
        
    int deflateInit2(z_stream* strm, int level, int method, int windowBits, int memLevel, int strategy)
    int deflate(z_stream* strm, int flush)
    int deflateEnd(z_stream* strm)
    
    int Z_OK
    int Z_FINISH
    int Z_STREAM_END
    int Z_DEFLATED
    int Z_DEFAULT_STRATEGY
    
def fast_compress(bytes data):
    """Fast gzip compression using direct zlib C API."""
    cdef:
        z_stream stream
        int ret
        unsigned char* input_data = data
        unsigned int input_len = len(data)
        unsigned int output_len = input_len + (input_len >> 12) + (input_len >> 14) + (input_len >> 25) + 13 + 18
        unsigned char* output_buffer = <unsigned char*>malloc(output_len)
        
    if output_buffer == NULL:
        raise MemoryError("Failed to allocate output buffer")
    
    # Initialize z_stream structure to zero
    stream.next_in = NULL
    stream.avail_in = 0
    stream.total_in = 0
    stream.next_out = NULL
    stream.avail_out = 0
    stream.total_out = 0
        
    stream.next_in = input_data
    stream.avail_in = input_len
    stream.next_out = output_buffer
    stream.avail_out = output_len
    
    # Initialize with gzip format (windowBits = 16 + 15)
    ret = deflateInit2(&stream, 9, Z_DEFLATED, 31, 8, Z_DEFAULT_STRATEGY)
    if ret != Z_OK:
        free(output_buffer)
        raise RuntimeError(f"deflateInit2 failed with error {ret}")
        
    ret = deflate(&stream, Z_FINISH)
    if ret != Z_STREAM_END:
        deflateEnd(&stream)
        free(output_buffer)
        raise RuntimeError(f"deflate failed with error {ret}")
        
    deflateEnd(&stream)
    
    # Create Python bytes object from compressed data
    result = PyBytes_FromStringAndSize(<char*>output_buffer, stream.total_out)
    free(output_buffer)
    
    return result