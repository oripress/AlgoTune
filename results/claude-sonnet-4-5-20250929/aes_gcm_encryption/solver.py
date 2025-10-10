from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Encrypt the plaintext using AES-GCM.
        
        Args:
            problem (dict): Dictionary with keys 'key', 'nonce', 'plaintext', 'associated_data'
        
        Returns:
            dict: Dictionary containing 'ciphertext' and 'tag'
        """
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]
        
        # Create AESGCM instance and encrypt
        aesgcm = AESGCM(key)
        ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, associated_data)
        
        # Split ciphertext and tag (tag is last 16 bytes)
        return {
            "ciphertext": ciphertext_with_tag[:-16],
            "tag": ciphertext_with_tag[-16:]
        }