import ctypes
import ctypes.util
from typing import Any, Callable, Dict

# This module is the final attempt to build the fastest possible SHA-256 solver.
# The core strategy is to call OpenSSL's one-shot EVP_Digest function directly
# via ctypes. This is theoretically faster than hashlib's two-step C process.
# This implementation includes aggressive micro-optimizations to minimize all
# Python and FFI overhead.

_solve_func: Callable[[bytes], bytes]

try:
    # Find and load libcrypto. Using CDLL(None) is a robust fallback for
    # sandboxed environments where hashlib has already loaded the library.
    libcrypto_path = ctypes.util.find_library('crypto')
    libcrypto = ctypes.CDLL(libcrypto_path or None)

    # Define C function signatures for performance and correctness.
    EVP_get_digestbyname = libcrypto.EVP_get_digestbyname
    EVP_get_digestbyname.argtypes = [ctypes.c_char_p]
    EVP_get_digestbyname.restype = ctypes.c_void_p

    EVP_Digest = libcrypto.EVP_Digest
    EVP_Digest.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p, ctypes.c_void_p
    ]
    EVP_Digest.restype = ctypes.c_int

    # --- One-time setup at module load time ---
    SHA256_MD_PTR = EVP_get_digestbyname(b"sha256")
    if not SHA256_MD_PTR:
        raise ImportError("SHA256 not supported by OpenSSL")

    # Pre-allocate all objects needed in the hot loop to avoid runtime overhead.
    digest_buffer = ctypes.create_string_buffer(32)
    digest_len = ctypes.c_uint()
    digest_len_ptr = ctypes.byref(digest_len) # Pre-create pointer object.

    def solve_with_ctypes(plaintext: bytes) -> bytes:
        """
        Calls the one-shot EVP_Digest C function. This is the hot path, stripped
        of all non-essential operations, including error checking.
        """
        EVP_Digest(
            plaintext, len(plaintext), digest_buffer,
            digest_len_ptr, SHA256_MD_PTR, None
        )
        # Return a new Python bytes object by slicing the raw buffer view.
        return digest_buffer.raw[:digest_len.value]

    _solve_func = solve_with_ctypes
    _solve_func(b"warmup") # Sanity check the FFI linkage.

except (ImportError, OSError, AttributeError):
    # Fallback to standard hashlib if the ctypes approach fails for any reason.
    import hashlib
    _sha256 = hashlib.sha256
    def solve_with_hashlib(plaintext: bytes) -> bytes:
        return _sha256(plaintext).digest()
    _solve_func = solve_with_hashlib

class Solver:
    """
    A solver that computes SHA-256 hashes by dispatching to the fastest
    available function, determined at module load time.
    """
    @staticmethod
    def solve(problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Computes the SHA-256 hash. This is a static method to eliminate
        instance binding overhead. It calls a pre-compiled function that
        uses the most optimized path available.
        """
        return {"digest": _solve_func(problem["plaintext"])}