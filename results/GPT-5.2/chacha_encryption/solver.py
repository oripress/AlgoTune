from __future__ import annotations

import struct
from typing import Any

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.poly1305 import Poly1305

_Z16 = b"\x00" * 16
_Z32 = b"\x00" * 32
_CTR0 = b"\x00\x00\x00\x00"
_CTR1 = b"\x01\x00\x00\x00"
_PACK_QQ = struct.Struct("<QQ").pack

class Solver:
    __slots__ = ()

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, bytes]:
        p = problem
        key: bytes = p["key"]
        nonce12: bytes = p["nonce"]
        plaintext: bytes = p["plaintext"]
        aad = p["associated_data"]
        aad_b: bytes = aad or b""

        # For small messages, OpenSSL's integrated AEAD is usually fastest.
        if len(plaintext) <= 2048:
            out = ChaCha20Poly1305(key).encrypt(nonce12, plaintext, aad_b)
            return {"ciphertext": out[:-16], "tag": out[-16:]}

        # Manual RFC8439 ChaCha20-Poly1305 to avoid extra copy from slicing
        # ciphertext||tag returned by ChaCha20Poly1305.encrypt().
        Cipher_ = Cipher
        ChaCha20 = algorithms.ChaCha20
        Poly = Poly1305

        nonce0 = _CTR0 + nonce12  # counter = 0
        nonce1 = _CTR1 + nonce12  # counter = 1

        poly_key = Cipher_(ChaCha20(key, nonce0), mode=None).encryptor().update(_Z32)
        poly = Poly(poly_key)

        ciphertext = Cipher_(ChaCha20(key, nonce1), mode=None).encryptor().update(plaintext)

        poly.update(aad_b)
        rem = len(aad_b) & 15
        if rem:
            poly.update(_Z16[: 16 - rem])

        poly.update(ciphertext)
        rem = len(ciphertext) & 15
        if rem:
            poly.update(_Z16[: 16 - rem])

        poly.update(_PACK_QQ(len(aad_b), len(ciphertext)))
        tag = poly.finalize()

        return {"ciphertext": ciphertext, "tag": tag}