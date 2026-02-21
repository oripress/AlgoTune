from typing import Any
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

class Solver:
    def __init__(self):
        self._aes_cache = {}

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, bytes]:
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem.get("associated_data")

        aes_algo = self._aes_cache.get(key)
        if aes_algo is None:
            aes_algo = algorithms.AES(key)
            self._aes_cache[key] = aes_algo

        encryptor = Cipher(aes_algo, modes.GCM(nonce)).encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
            
        ciphertext = encryptor.update(plaintext)
        encryptor.finalize()
        
        return {
            "ciphertext": ciphertext,
            "tag": encryptor.tag
        }