# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False

from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t
from libc.string cimport memcpy

cdef extern from "openssl/evp.h":
    ctypedef struct EVP_CIPHER
    ctypedef struct EVP_CIPHER_CTX

    EVP_CIPHER_CTX *EVP_CIPHER_CTX_new()
    void EVP_CIPHER_CTX_free(EVP_CIPHER_CTX *c)
    const EVP_CIPHER *EVP_chacha20_poly1305()
    int EVP_EncryptInit_ex(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *type, void *impl, const unsigned char *key, const unsigned char *iv)
    int EVP_EncryptUpdate(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outlen, const unsigned char *in_, int inlen)
    int EVP_EncryptFinal_ex(EVP_CIPHER_CTX *ctx, unsigned char *outm, int *outlen)
    int EVP_CIPHER_CTX_ctrl(EVP_CIPHER_CTX *ctx, int type, int arg, void *ptr)

cdef int EVP_CTRL_AEAD_SET_IVLEN = 0x9
cdef int EVP_CTRL_AEAD_GET_TAG = 0x10
cdef int TAG_SIZE = 16

def encrypt(bytes key, bytes nonce, bytes plaintext, associated_data):
    cdef bytes aad
    if associated_data is None:
        aad = b""
    else:
        aad = associated_data

    cdef Py_ssize_t pt_len = len(plaintext)
    cdef Py_ssize_t aad_len = len(aad)
    cdef EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new()
    cdef int outlen = 0
    cdef int tmp = 0
    cdef bytes ciphertext
    cdef bytes tag
    cdef unsigned char *ct_ptr
    cdef unsigned char *tag_ptr

    if ctx == NULL:
        raise MemoryError()

    try:
        if len(key) != 32:
            raise ValueError("Invalid key size")
        if len(nonce) != 12:
            raise ValueError("Invalid nonce size")

        if EVP_EncryptInit_ex(ctx, EVP_chacha20_poly1305(), NULL, NULL, NULL) != 1:
            raise ValueError("EVP_EncryptInit_ex failed")
        if EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_IVLEN, 12, NULL) != 1:
            raise ValueError("EVP_CIPHER_CTX_ctrl set ivlen failed")
        if EVP_EncryptInit_ex(
            ctx,
            NULL,
            NULL,
            <const unsigned char *>PyBytes_AS_STRING(key),
            <const unsigned char *>PyBytes_AS_STRING(nonce),
        ) != 1:
            raise ValueError("EVP_EncryptInit_ex key/iv failed")

        if aad_len:
            if EVP_EncryptUpdate(
                ctx,
                NULL,
                &tmp,
                <const unsigned char *>PyBytes_AS_STRING(aad),
                <int>aad_len,
            ) != 1:
                raise ValueError("AAD update failed")

        ciphertext = PyBytes_FromStringAndSize(NULL, pt_len)
        if ciphertext is None:
            raise MemoryError()
        ct_ptr = <unsigned char *>PyBytes_AS_STRING(ciphertext)

        if pt_len:
            if EVP_EncryptUpdate(
                ctx,
                ct_ptr,
                &outlen,
                <const unsigned char *>PyBytes_AS_STRING(plaintext),
                <int>pt_len,
            ) != 1:
                raise ValueError("Encrypt update failed")
        else:
            outlen = 0

        if EVP_EncryptFinal_ex(ctx, ct_ptr + outlen, &tmp) != 1:
            raise ValueError("Encrypt final failed")

        tag = PyBytes_FromStringAndSize(NULL, TAG_SIZE)
        if tag is None:
            raise MemoryError()
        tag_ptr = <unsigned char *>PyBytes_AS_STRING(tag)
        if EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_GET_TAG, TAG_SIZE, tag_ptr) != 1:
            raise ValueError("Get tag failed")

        return {"ciphertext": ciphertext, "tag": tag}
    finally:
        EVP_CIPHER_CTX_free(ctx)