from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers.algorithms import AES
from cryptography.hazmat.primitives.ciphers.modes import GCM

class Solver:
    __slots__ = ()
    
    def solve(self, problem, _Cipher=Cipher, _AES=AES, _GCM=GCM):
        p = problem
        e = _Cipher(_AES(p["key"]), _GCM(p["nonce"])).encryptor()
        ad = p["associated_data"]
        if ad:
            e.authenticate_additional_data(ad)
        ct = e.update(p["plaintext"])
        e.finalize()
        return {"ciphertext": ct, "tag": e.tag}