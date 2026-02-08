from typing import Any
import cffi

GCM_TAG_SIZE = 16

_ffi = cffi.FFI()
_ffi.cdef("""
    typedef ... EVP_CIPHER_CTX;
    typedef ... EVP_CIPHER;
    typedef ... ENGINE;
    EVP_CIPHER_CTX *EVP_CIPHER_CTX_new(void);
    void EVP_CIPHER_CTX_free(EVP_CIPHER_CTX *ctx);
    int EVP_CIPHER_CTX_reset(EVP_CIPHER_CTX *ctx);
    const EVP_CIPHER *EVP_aes_128_gcm(void);
    const EVP_CIPHER *EVP_aes_192_gcm(void);
    const EVP_CIPHER *EVP_aes_256_gcm(void);
    int EVP_EncryptInit_ex(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *type,
                           ENGINE *impl, const unsigned char *key,
                           const unsigned char *iv);
    int EVP_EncryptUpdate(EVP_CIPHER_CTX *ctx, unsigned char *out,
                          int *outl, const unsigned char *in, int inl);
    int EVP_EncryptFinal_ex(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl);
    int EVP_CIPHER_CTX_ctrl(EVP_CIPHER_CTX *ctx, int type, int arg, void *ptr);
""")

try:
    _lib = _ffi.dlopen("libcrypto.so")
except OSError:
    try:
        _lib = _ffi.dlopen("libcrypto.so.3")
    except OSError:
        _lib = _ffi.dlopen("libcrypto.so.1.1")

_NULL = _ffi.NULL
_AES_128_GCM = _lib.EVP_aes_128_gcm()
_AES_192_GCM = _lib.EVP_aes_192_gcm()
_AES_256_GCM = _lib.EVP_aes_256_gcm()
_CIPHER_MAP = {16: _AES_128_GCM, 24: _AES_192_GCM, 32: _AES_256_GCM}

EVP_CTRL_GCM_SET_IVLEN = 0x9
EVP_CTRL_GCM_GET_TAG = 0x10

_EncryptInit = _lib.EVP_EncryptInit_ex
_EncryptUpdate = _lib.EVP_EncryptUpdate
_EncryptFinal = _lib.EVP_EncryptFinal_ex
_CTX_ctrl = _lib.EVP_CIPHER_CTX_ctrl
_CTX_reset = _lib.EVP_CIPHER_CTX_reset
_buffer = _ffi.buffer
_new = _ffi.new

class Solver:
    def __init__(self):
        self._ctx = _lib.EVP_CIPHER_CTX_new()
        self._outlen = _new("int *")
        self._tag_buf = _new("unsigned char[]", GCM_TAG_SIZE)
        self._ct_buf = None
        self._ct_buf_size = 0

    def solve(self, problem: dict, **kwargs) -> Any:
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        aad = problem["associated_data"]

        pt_len = len(plaintext)
        ctx = self._ctx
        outlen = self._outlen

        _CTX_reset(ctx)
        key_len = len(key)
        nonce_len = len(nonce)
        if nonce_len != 12:
            _EncryptInit(ctx, _CIPHER_MAP[key_len], _NULL, _NULL, _NULL)
            _CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, nonce_len, _NULL)
            _EncryptInit(ctx, _NULL, _NULL, key, nonce)
        else:
            _EncryptInit(ctx, _CIPHER_MAP[key_len], _NULL, key, nonce)

        if aad:
            _EncryptUpdate(ctx, _NULL, outlen, aad, len(aad))

        needed = pt_len + 16
        if needed > self._ct_buf_size:
            self._ct_buf = _new("unsigned char[]", needed)
            self._ct_buf_size = needed
        ct_buf = self._ct_buf

        _EncryptUpdate(ctx, ct_buf, outlen, plaintext, pt_len)
        written = outlen[0]

        _EncryptFinal(ctx, ct_buf + written, outlen)
        written += outlen[0]

        tag_buf = self._tag_buf
        _CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, GCM_TAG_SIZE, tag_buf)

        return {
            "ciphertext": _buffer(ct_buf, written)[:],
            "tag": _buffer(tag_buf, GCM_TAG_SIZE)[:]
        }
        return {"ciphertext": ciphertext, "tag": tag}