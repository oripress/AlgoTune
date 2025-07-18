# distutils: language = c
# distutils: libraries = crypto

cimport cython
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint8_t, size_t

cdef extern from "openssl/chacha.h":
    int CRYPTO_chacha20_poly1305_ietf_encrypt(const uint8_t *in, size_t in_len,
                                              const uint8_t *ad, size_t ad_len,
                                              const uint8_t *nsec,
                                              const uint8_t *nonce,
                                              const uint8_t *key,
                                              uint8_t *out,
                                              uint8_t *tag)

@cython.boundscheck(False)
@cython.wraparound(False)
def encrypt_chacha20poly1305(bytes key, bytes nonce, bytes plaintext, bytes aad=None):
    cdef const uint8_t *in_buf = <const uint8_t *> plaintext
    cdef size_t in_len = len(plaintext)
    cdef uint8_t *out_buf = <uint8_t *> malloc(in_len)
    if out_buf == NULL:
        raise MemoryError()
    cdef uint8_t tag_buf[16]
    cdef const uint8_t *ad_buf = NULL
    cdef size_t ad_len = 0
    if aad:
        ad_buf = <const uint8_t *> aad
        ad_len = len(aad)
    cdef const uint8_t *nsec = NULL
    cdef int rc = CRYPTO_chacha20_poly1305_ietf_encrypt(
        in_buf, in_len, ad_buf, ad_len, nsec,
        <const uint8_t *> nonce, <const uint8_t *> key,
        out_buf, tag_buf
    )
    if rc != 1:
        free(out_buf)
        raise RuntimeError("ChaCha20-Poly1305 encryption failed")
    cdef bytes ciphertext = PyBytes_FromStringAndSize(<char *> out_buf, in_len)
    cdef bytes tag = PyBytes_FromStringAndSize(<char *> tag_buf, 16)
    free(out_buf)
    return ciphertext, tag