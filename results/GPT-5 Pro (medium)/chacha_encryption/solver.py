from typing import Any, Dict
from functools import lru_cache
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

_TAG_SIZE = 16

@lru_cache(maxsize=64)
def _get_cipher(key: bytes) -> ChaCha20Poly1305:
    return ChaCha20Poly1305(key)

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, bytes]:
        """
        Encrypt the provided plaintext using ChaCha20-Poly1305 and return
        the ciphertext and authentication tag separately.
        """
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]

        encrypt = _get_cipher(key).encrypt
        out = encrypt(nonce, plaintext, associated_data)

        return {"ciphertext": out[:-_TAG_SIZE], "tag": out[-_TAG_SIZE:]}