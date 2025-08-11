from typing import Any
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

CHACHA_KEY_SIZE = 32  # ChaCha20 uses 256-bit keys
POLY1305_TAG_SIZE = 16  # Poly1305 produces 128-bit tags

class Solver:
    def __init__(self):
        self._cipher_cache = {}
        
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Solve the ChaCha20-Poly1305 encryption problem by encrypting the plaintext.
        Optimized version with cipher caching.
        """
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]
        
        # Cache the cipher object based on the key
        key_id = id(key)  # Use object id for fast comparison
        if key_id not in self._cipher_cache:
            self._cipher_cache[key_id] = ChaCha20Poly1305(key)
        
        chacha = self._cipher_cache[key_id]
        ciphertext = chacha.encrypt(nonce, plaintext, associated_data)
        
        # Direct slicing without intermediate variables
        return {
            "ciphertext": ciphertext[:-POLY1305_TAG_SIZE],
            "tag": ciphertext[-POLY1305_TAG_SIZE:]
        }