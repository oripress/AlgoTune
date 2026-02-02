from typing import Any
from cryptography.hazmat.primitives import hashes

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Compute the SHA-256 hash of the plaintext using the cryptography library.
        """
        plaintext = problem["plaintext"]

        try:
            digest = hashes.Hash(hashes.SHA256())
            digest.update(plaintext)
            hash_value = digest.finalize()

            return {"digest": hash_value}

        except Exception as e:
            raise