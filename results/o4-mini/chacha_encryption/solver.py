import ast
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

CHACHA_KEY_SIZE = 32
POLY1305_TAG_SIZE = 16

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        # Parse input if it's a string literal
        if isinstance(problem, str):
            problem = ast.literal_eval(problem)
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem.get("associated_data") or b''
        # Validate sizes
        if len(key) != CHACHA_KEY_SIZE:
            raise ValueError(f"Invalid key size: {len(key)}. Must be {CHACHA_KEY_SIZE}.")
        chacha = ChaCha20Poly1305(key)
        ct_and_tag = chacha.encrypt(nonce, plaintext, associated_data)
        if len(ct_and_tag) < POLY1305_TAG_SIZE:
            raise ValueError("Encrypted output is shorter than the expected tag size.")
        return {
            "ciphertext": ct_and_tag[:-POLY1305_TAG_SIZE],
            "tag": ct_and_tag[-POLY1305_TAG_SIZE:]
        }