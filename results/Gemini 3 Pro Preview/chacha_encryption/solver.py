import ctypes
import ctypes.util
import sys
from typing import Any

# Constants
EVP_CTRL_AEAD_SET_IVLEN = 0x9
EVP_CTRL_AEAD_GET_TAG = 0x10

# Load Library
_lib_path = ctypes.util.find_library('crypto')
_lib = ctypes.CDLL(_lib_path) if _lib_path else None

if _lib:
    # Define types
    c_void_p = ctypes.c_void_p
    c_int = ctypes.c_int
    c_char_p = ctypes.c_char_p
    c_ssize_t = ctypes.c_ssize_t
    py_object = ctypes.py_object

    # OpenSSL Functions
    _lib.EVP_CIPHER_CTX_new.restype = c_void_p
    _lib.EVP_CIPHER_CTX_new.argtypes = []

    _lib.EVP_CIPHER_CTX_free.restype = None
    _lib.EVP_CIPHER_CTX_free.argtypes = [c_void_p]

    _lib.EVP_chacha20_poly1305.restype = c_void_p
    _lib.EVP_chacha20_poly1305.argtypes = []

    _lib.EVP_EncryptInit_ex.restype = c_int
    _lib.EVP_EncryptInit_ex.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]

    _lib.EVP_CIPHER_CTX_ctrl.restype = c_int
    _lib.EVP_CIPHER_CTX_ctrl.argtypes = [c_void_p, c_int, c_int, c_void_p]

    _lib.EVP_EncryptUpdate.restype = c_int
    _lib.EVP_EncryptUpdate.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int]

    _lib.EVP_EncryptFinal_ex.restype = c_int
    _lib.EVP_EncryptFinal_ex.argtypes = [c_void_p, c_void_p, c_void_p]

    # Python C-API
    _pyapi = ctypes.pythonapi
    _pyapi.PyBytes_FromStringAndSize.restype = py_object
    _pyapi.PyBytes_FromStringAndSize.argtypes = [c_char_p, c_ssize_t]

    _pyapi.PyBytes_AsString.restype = c_void_p
    _pyapi.PyBytes_AsString.argtypes = [py_object]

    class Solver:
        def __init__(self):
            self.ctx = _lib.EVP_CIPHER_CTX_new()
            if not self.ctx:
                raise MemoryError("Failed to create EVP_CIPHER_CTX")
            self.cipher = _lib.EVP_chacha20_poly1305()
            self.outlen = ctypes.c_int(0)
            self.outlen_ptr = ctypes.byref(self.outlen)
            self.null = None
            
            # Pre-initialize context with cipher
            if _lib.EVP_EncryptInit_ex(self.ctx, self.cipher, self.null, self.null, self.null) != 1:
                raise RuntimeError("EVP_EncryptInit_ex failed")

        def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
            key = problem["key"]
            nonce = problem["nonce"]
            plaintext = problem["plaintext"]
            aad = problem["associated_data"]
            
            ctx = self.ctx
            
            # 1. Init context with cipher
            if _lib.EVP_EncryptInit_ex(ctx, self.cipher, self.null, self.null, self.null) != 1:
                raise RuntimeError("EVP_EncryptInit_ex failed")

            # 2. Set IV length to 12
            if _lib.EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_IVLEN, 12, self.null) != 1:
                raise RuntimeError("Set IV len failed")

            # 3. Set Key and Nonce
            key_ptr = _pyapi.PyBytes_AsString(key)
            nonce_ptr = _pyapi.PyBytes_AsString(nonce)
            
            if _lib.EVP_EncryptInit_ex(ctx, self.null, self.null, key_ptr, nonce_ptr) != 1:
                raise RuntimeError("Set Key/Nonce failed")

            # 4. Process AAD
            if aad:
                aad_len = len(aad)
                aad_ptr = _pyapi.PyBytes_AsString(aad)
                if _lib.EVP_EncryptUpdate(ctx, self.null, self.outlen_ptr, aad_ptr, aad_len) != 1:
                    raise RuntimeError("AAD update failed")

            # 5. Process Plaintext
            pt_len = len(plaintext)
            pt_ptr = _pyapi.PyBytes_AsString(plaintext)
            
            # Allocate output ciphertext
            ciphertext = _pyapi.PyBytes_FromStringAndSize(None, pt_len)
            ct_ptr = _pyapi.PyBytes_AsString(ciphertext)
            
            if _lib.EVP_EncryptUpdate(ctx, ct_ptr, self.outlen_ptr, pt_ptr, pt_len) != 1:
                raise RuntimeError("EncryptUpdate failed")
                
            # 6. Finalize
            if _lib.EVP_EncryptFinal_ex(ctx, self.null, self.outlen_ptr) != 1:
                raise RuntimeError("EncryptFinal failed")
                
            # 7. Get Tag
            tag = _pyapi.PyBytes_FromStringAndSize(None, 16)
            tag_ptr = _pyapi.PyBytes_AsString(tag)
            
            if _lib.EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_GET_TAG, 16, tag_ptr) != 1:
                raise RuntimeError("Get Tag failed")
                
            return {"ciphertext": ciphertext, "tag": tag}

else:
    # Fallback
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    class Solver:
        def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
            key = problem["key"]
            nonce = problem["nonce"]
            plaintext = problem["plaintext"]
            associated_data = problem["associated_data"]
            chacha = ChaCha20Poly1305(key)
            full = chacha.encrypt(nonce, plaintext, associated_data)
            return {"ciphertext": full[:-16], "tag": full[-16:]}