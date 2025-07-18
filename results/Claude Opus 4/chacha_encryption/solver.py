from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from typing import Any

CHACHA_KEY_SIZE = 32
POLY1305_TAG_SIZE = 16

class Solver:
    def __init__(self):
        self._cipher_cache = {}
    
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the ChaCha20-Poly1305 encryption problem by encrypting the plaintext.
        """
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]
        
        if len(key) != CHACHA_KEY_SIZE:
            raise ValueError(f"Invalid key size: {len(key)}. Must be {CHACHA_KEY_SIZE}.")
        
        # Cache cipher objects by key
        if key not in self._cipher_cache:
            self._cipher_cache[key] = ChaCha20Poly1305(key)
        
        chacha = self._cipher_cache[key]
        ciphertext = chacha.encrypt(nonce, plaintext, associated_data)
        
        # Extract ciphertext and tag
        actual_ciphertext = ciphertext[:-POLY1305_TAG_SIZE]
        tag = ciphertext[-POLY1305_TAG_SIZE:]
        
        return {"ciphertext": actual_ciphertext, "tag": tag}