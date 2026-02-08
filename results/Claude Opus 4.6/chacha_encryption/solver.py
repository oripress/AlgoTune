from typing import Any
import cffi

POLY1305_TAG_SIZE = 16

_ffi = cffi.FFI()
_ffi.cdef("""
    typedef ... EVP_CIPHER_CTX;
    typedef ... EVP_CIPHER;
    typedef ... ENGINE;
    
    EVP_CIPHER_CTX *EVP_CIPHER_CTX_new(void);
    void EVP_CIPHER_CTX_free(EVP_CIPHER_CTX *ctx);
    const EVP_CIPHER *EVP_chacha20_poly1305(void);
    int EVP_EncryptInit_ex(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *type,
                           ENGINE *impl, const unsigned char *key, const unsigned char *iv);
    int EVP_EncryptUpdate(EVP_CIPHER_CTX *ctx, unsigned char *out,
                          int *outl, const unsigned char *in, int inl);
    int EVP_EncryptFinal_ex(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl);
    int EVP_CIPHER_CTX_ctrl(EVP_CIPHER_CTX *ctx, int type, int arg, void *ptr);
""")

_lib = None
for _name in ['libcrypto.so.3', 'libcrypto.so.1.1', 'libcrypto.so']:
    try:
        _lib = _ffi.dlopen(_name)
        break
    except OSError:
        continue

_USE_CFFI = _lib is not None

if _USE_CFFI:
    _cipher_ptr = _lib.EVP_chacha20_poly1305()
    _EVP_CTRL_AEAD_SET_IVLEN = 0x9
    _EVP_CTRL_AEAD_GET_TAG = 0x10
    _NULL_CTX = _ffi.NULL
else:
    _cipher_ptr = None
    _EVP_CTRL_AEAD_SET_IVLEN = 0
    _EVP_CTRL_AEAD_GET_TAG = 0
    _NULL_CTX = None

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, bytes]:
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        aad = problem["associated_data"]

        if _USE_CFFI:
            lib = _lib
            ffi = _ffi
            
            ctx = lib.EVP_CIPHER_CTX_new()
            try:
                lib.EVP_EncryptInit_ex(ctx, _cipher_ptr, _NULL_CTX, _NULL_CTX, _NULL_CTX)
                lib.EVP_CIPHER_CTX_ctrl(ctx, _EVP_CTRL_AEAD_SET_IVLEN, 12, _NULL_CTX)
                lib.EVP_EncryptInit_ex(ctx, _NULL_CTX, _NULL_CTX, key, nonce)

                outlen = ffi.new("int *")

                if aad:
                    lib.EVP_EncryptUpdate(ctx, _NULL_CTX, outlen, aad, len(aad))

                pt_len = len(plaintext)
                ct_buf = ffi.new("unsigned char[]", pt_len + 16)
                lib.EVP_EncryptUpdate(ctx, ct_buf, outlen, plaintext, pt_len)
                ct_written = outlen[0]

                lib.EVP_EncryptFinal_ex(ctx, ct_buf + ct_written, outlen)

                tag_buf = ffi.new("unsigned char[]", 16)
                lib.EVP_CIPHER_CTX_ctrl(ctx, _EVP_CTRL_AEAD_GET_TAG, 16, tag_buf)

                return {
                    "ciphertext": ffi.buffer(ct_buf, ct_written)[:],
                    "tag": ffi.buffer(tag_buf, 16)[:]
                }
            finally:
                lib.EVP_CIPHER_CTX_free(ctx)
        else:
            chacha = ChaCha20Poly1305(key)
            ciphertext = chacha.encrypt(nonce, plaintext, aad)
            return {
                "ciphertext": ciphertext[:-POLY1305_TAG_SIZE],
                "tag": ciphertext[-POLY1305_TAG_SIZE:]
            }