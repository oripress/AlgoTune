from __future__ import annotations

from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class Solver:
    __slots__ = ("_encrypt_cache",)

    def __init__(self) -> None:
        self._encrypt_cache: dict[bytes, Any] = {}

    def solve(self, problem, **kwargs) -> Any:
        cache = self._encrypt_cache
        key = problem["key"]
        encrypt = cache.get(key)
        if encrypt is None:
            encrypt = AESGCM(key).encrypt
            cache[key] = encrypt
        encrypted = encrypt(
            problem["nonce"], problem["plaintext"], problem["associated_data"]
        )
        return {"ciphertext": encrypted[:-16], "tag": encrypted[-16:]}