from typing import Any
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        # Extract values
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]

        # Encrypt using AES-GCM
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)

        # Split ciphertext and tag (GCM tag is 16 bytes)
        tag = ciphertext[-16:]
        ciphertext_only = ciphertext[:-16]

        return {"ciphertext": ciphertext_only, "tag": tag}