from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class Solver:
    def solve(self, problem):
        key, nonce, plaintext, aad = problem["key"], problem["nonce"], problem["plaintext"], problem["associated_data"]
        if aad is None:
            aad = b''
        
        aesgcm = AESGCM(key)
        result = aesgcm.encrypt(nonce, plaintext, aad)
        tag_start = len(result) - 16
        return {
            "ciphertext": result[:tag_start],
            "tag": result[tag_start:]
        }