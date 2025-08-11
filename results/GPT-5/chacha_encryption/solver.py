from typing import Any
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

class Solver:
    __slots__ = ("_enc_cache",)

    def __init__(self) -> None:
        # Cache bound encrypt method per key to avoid re-instantiation and lookups
        self._enc_cache: dict[bytes, Any] = {}

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, bytes]:
        p = problem
        key = p["key"]
        enc = self._enc_cache.get(key)
        if enc is None:
            enc = ChaCha20Poly1305(key).encrypt
            self._enc_cache[key] = enc

        ct = enc(p["nonce"], p["plaintext"], p.get("associated_data", None))
        return {"ciphertext": ct[:-16], "tag": ct[-16:]}