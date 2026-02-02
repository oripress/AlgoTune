from __future__ import annotations

from typing import Any

from cryptography.hazmat.primitives.ciphers import Cipher as _Cipher, algorithms as _algorithms, modes as _modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _AESGCM

_AES = _algorithms.AES
_GCM = _modes.GCM

_TAG_SIZE = 16

# Tuneable cutoff: below this, AESGCM() tends to have lower Python-side overhead;
# above this, avoiding an extra O(n) slice/copy wins.
_SMALL_PLAINTEXT_CUTOFF = 2048

class Solver:
    __slots__ = ()

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        key = problem["key"]
        nonce = problem["nonce"]
        pt = problem["plaintext"]
        aad = problem["associated_data"]

        if len(pt) <= _SMALL_PLAINTEXT_CUTOFF:
            out = _AESGCM(key).encrypt(nonce, pt, aad)
            t = _TAG_SIZE
            return {"ciphertext": out[:-t], "tag": out[-t:]}

        encryptor = _Cipher(_AES(key), _GCM(nonce)).encryptor()
        if aad:
            encryptor.authenticate_additional_data(aad)
        ct = encryptor.update(pt)
        encryptor.finalize()
        return {"ciphertext": ct, "tag": encryptor.tag}