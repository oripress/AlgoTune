from typing import Any
from cryptography.hazmat.primitives.ciphers import algorithms, modes
from cryptography.hazmat.backends import default_backend

AES_KEY_SIZES = {16, 24, 32}
BACKEND = default_backend()

class Solver:
    def __init__(self):
        self._aes_cache = {}

    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]

        # Validate key size
        if len(key) not in AES_KEY_SIZES:
            raise ValueError(f"Invalid key size: {len(key)}. Must be one of {AES_KEY_SIZES}.")

        # Get AES algorithm object
        algo = self._aes_cache.get(key)
        if algo is None:
            algo = algorithms.AES(key)
            # Simple cache limit
            if len(self._aes_cache) < 128:
                self._aes_cache[key] = algo

        # Create Encryptor directly from backend to avoid Cipher object overhead
        encryptor = BACKEND.create_symmetric_encryption_ctx(algo, modes.GCM(nonce))
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
            
        # update returns ciphertext, finalize returns empty bytes for GCM
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        return {"ciphertext": ciphertext, "tag": encryptor.tag}