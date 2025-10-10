from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from typing import Any

class Solver:
    def __init__(self):
        # Cache ChaCha20Poly1305 objects by key to avoid repeated initialization
        self._cipher_cache = {}
    
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Solve the ChaCha20-Poly1305 encryption problem by encrypting the plaintext.
        Optimized version with cipher caching and minimal overhead.
        """
        key = problem["key"]
        
        # Get or create cached cipher object
        if key not in self._cipher_cache:
            self._cipher_cache[key] = ChaCha20Poly1305(key)
        
        # Perform encryption
        ciphertext = self._cipher_cache[key].encrypt(
            problem["nonce"], 
            problem["plaintext"], 
            problem["associated_data"]
        )
        
        # Split ciphertext and tag (tag is last 16 bytes)
        return {"ciphertext": ciphertext[:-16], "tag": ciphertext[-16:]}