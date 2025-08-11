from typing import Any
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

# Constants for ChaCha20-Poly1305
CHACHA_KEY_SIZE = 32
POLY1305_TAG_SIZE = 16

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Solve the ChaCha20-Poly1305 encryption problem by encrypting the plaintext.
        Uses cryptography.hazmat.primitives.ciphers.aead.ChaCha20Poly1305 to compute:
            ciphertext, tag = encrypt(key, nonce, plaintext, associated_data)

        :param problem: A dictionary containing the encryption inputs.
        :return: A dictionary with keys:
                 "ciphertext": The encrypted data
                 "tag": The authentication tag (16 bytes)
        """
        # Extract inputs
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]

        # Validate key size
        if len(key) != CHACHA_KEY_SIZE:
            raise ValueError(f"Invalid key size: {len(key)}. Must be {CHACHA_KEY_SIZE}.")

        # Create cipher and encrypt in one line
        ciphertext = ChaCha20Poly1305(key).encrypt(nonce, plaintext, associated_data)

        # Extract results
        tag = ciphertext[-POLY1305_TAG_SIZE:]
        actual_ciphertext = ciphertext[:-POLY1305_TAG_SIZE]

        return {"ciphertext": actual_ciphertext, "tag": tag}