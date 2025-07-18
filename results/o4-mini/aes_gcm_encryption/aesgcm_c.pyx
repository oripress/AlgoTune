# distutils: language = c

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdlib cimport malloc, free

cdef extern from "openssl/evp.h":
    ctypedef struct evp_cipher_ctx_st EVP_CIPHER_CTX
    ctypedef struct evp_cipher_st EVP_CIPHER
    EVP_CIPHER_CTX *EVP_CIPHER_CTX_new()
    void EVP_CIPHER_CTX_free(EVP_CIPHER_CTX *ctx)
    EVP_CIPHER *EVP_aes_128_gcm()
    EVP_CIPHER *EVP_aes_192_gcm()
    EVP_CIPHER *EVP_aes_256_gcm()
    int EVP_EncryptInit_ex(EVP_CIPHER_CTX *ctx, EVP_CIPHER *cipher,
                           void *impl, const unsigned char *key,
                           const unsigned char *iv)
    int EVP_CIPHER_CTX_ctrl(EVP_CIPHER_CTX *ctx, int type, int arg, void *ptr)
    int EVP_EncryptUpdate(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl,
                          const unsigned char *in, int inl)
    int EVP_EncryptFinal_ex(EVP_CIPHER_CTX *ctx, unsigned char *outm, int *outl)

cdef int EVP_CTRL_GCM_GET_TAG = 0x10
cdef int EVP_CTRL_GCM_SET_TAG = 0x11

def aesgcm_encrypt(key, iv, pt, ad):
    """
    AES-GCM encryption wrapper using OpenSSL EVP via Cython.
    Returns (ciphertext_bytes, tag_bytes).
    """
    cdef EVP_CIPHER_CTX *ctx
    cdef EVP_CIPHER *cipher
    cdef int outl
    cdef unsigned char *outbuf
    cdef unsigned char tagbuf[16]
    cdef bytes ct, tag
    cdef int keylen = len(key)
    ctx = EVP_CIPHER_CTX_new()
    if ctx == NULL:
        raise MemoryError()
    # choose cipher based on key length
    if keylen == 16:
        cipher = EVP_aes_128_gcm()
    elif keylen == 24:
        cipher = EVP_aes_192_gcm()
    elif keylen == 32:
        cipher = EVP_aes_256_gcm()
    else:
        EVP_CIPHER_CTX_free(ctx)
        raise ValueError("Invalid AES key size")
    if not EVP_EncryptInit_ex(ctx, cipher, NULL, <unsigned char *>key, <unsigned char *>iv):
        EVP_CIPHER_CTX_free(ctx)
        raise Exception("EVP_EncryptInit_ex failed")
    if ad:
        if not EVP_EncryptUpdate(ctx, NULL, &outl, <unsigned char *>ad, len(ad)):
            EVP_CIPHER_CTX_free(ctx)
            raise Exception("EVP_EncryptUpdate(AAD) failed")
    # allocate output buffer for ciphertext
    outbuf = <unsigned char *>malloc(len(pt))
    if outbuf == NULL:
        EVP_CIPHER_CTX_free(ctx)
        raise MemoryError()
    if not EVP_EncryptUpdate(ctx, outbuf, &outl, <unsigned char *>pt, len(pt)):
        EVP_CIPHER_CTX_free(ctx)
        free(outbuf)
        raise Exception("EVP_EncryptUpdate failed")
    if not EVP_EncryptFinal_ex(ctx, NULL, &outl):
        EVP_CIPHER_CTX_free(ctx)
        free(outbuf)
        raise Exception("EVP_EncryptFinal_ex failed")
    if not EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, tagbuf):
        EVP_CIPHER_CTX_free(ctx)
        free(outbuf)
        raise Exception("EVP_CIPHER_CTX_ctrl(GET_TAG) failed")
    # build Python bytes objects
    ct = PyBytes_FromStringAndSize(<char *>outbuf, len(pt))
    tag = PyBytes_FromStringAndSize(<char *>tagbuf, 16)
    EVP_CIPHER_CTX_free(ctx)
    free(outbuf)
    return ct, tag