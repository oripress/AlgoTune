from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

class Solver:
    def solve(self, problem, **kwargs):
        # Create a ChaCha20Poly1305 cipher object and encrypt directly
        ciphertext_with_tag = ChaCha20Poly1305(problem["key"]).encrypt(
            problem["nonce"],
            problem["plaintext"],
            problem.get("associated_data", None)
        )
        
        # Split into ciphertext and tag
        return {
            "ciphertext": ciphertext_with_tag[:-16],
            "tag": ciphertext_with_tag[-16:]
        }
        return {"ciphertext": ciphertext, "tag": tag}