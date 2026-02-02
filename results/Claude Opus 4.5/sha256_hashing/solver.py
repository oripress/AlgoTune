from cffi import FFI

ffi = FFI()
ffi.cdef("""
    unsigned char *SHA256(const unsigned char *d, size_t n, unsigned char *md);
""")

# Load OpenSSL's crypto library
try:
    _lib = ffi.dlopen("libcrypto.so.3")
except OSError:
    try:
        _lib = ffi.dlopen("libcrypto.so.1.1")
    except OSError:
        _lib = ffi.dlopen("libcrypto.so")

class Solver:
    def __init__(self):
        self._lib = _lib
        self._buf = ffi.new("unsigned char[32]")
    
    def solve(self, problem, **kwargs):
        pt = problem["plaintext"]
        self._lib.SHA256(pt, len(pt), self._buf)
        return {"digest": ffi.buffer(self._buf)[:]}