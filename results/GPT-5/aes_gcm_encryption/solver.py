from typing import Any

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

AES = algorithms.AES
GCM = modes.GCM

class Solver:
    __slots__ = ()

    def __init__(self) -> None:
        pass

    def solve(self, problem, **kwargs) -> Any:
        p = problem
        key = p["key"]
        nonce = p["nonce"]
        plaintext = p["plaintext"]
        associated_data = p.get("associated_data")

        encryptor = Cipher(AES(key), GCM(nonce)).encryptor()
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)

        ciphertext = encryptor.update(plaintext)
        encryptor.finalize()
        tag = encryptor.tag

        return {"ciphertext": ciphertext, "tag": tag}