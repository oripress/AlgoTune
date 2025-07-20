from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from typing import Any

class Solver:
    def __init__(self):
        self._backend = default_backend()
    
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Encrypt the plaintext using AES-GCM using lower-level API.
        """
        # Direct access without intermediate variables
        cipher = Cipher(
            algorithms.AES(problem["key"]),
            modes.GCM(problem["nonce"]),
            backend=self._backend
        )
        
        encryptor = cipher.encryptor()
        
        # Add associated data if present
        if problem["associated_data"]:
            encryptor.authenticate_additional_data(problem["associated_data"])
        
        # Encrypt in one go
        ciphertext = encryptor.update(problem["plaintext"]) + encryptor.finalize()
        
        return {"ciphertext": ciphertext, "tag": encryptor.tag}