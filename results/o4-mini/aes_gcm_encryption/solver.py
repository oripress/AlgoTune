from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class Solver:
    def solve(self, p):
        key = p["key"]
        nonce = p["nonce"]
        plaintext = p["plaintext"]
        aad = p.get("associated_data", None)
        # Use low‐level AES‐GCM encryptor
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
        enc = cipher.encryptor()
        if aad:
            enc.authenticate_additional_data(aad)
        ct = enc.update(plaintext) + enc.finalize()
        return {"ciphertext": ct, "tag": enc.tag}