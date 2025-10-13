from typing import Any, Dict, Mapping
import hashlib

class Solver:
    def solve(self, problem: Mapping[str, Any], **kwargs) -> Dict[str, bytes]:
        """
        Compute the SHA-256 hash of the given plaintext.

        :param problem: A mapping containing the key "plaintext" with a bytes-like object.
        :return: A dict with key "digest" containing the 32-byte SHA-256 hash.
        """
        plaintext = problem["plaintext"]  # Intentionally let KeyError propagate if missing

        # Accept common bytes-like types similar to the reference behavior.
        if isinstance(plaintext, (bytes, bytearray)):
            data = plaintext
        elif isinstance(plaintext, memoryview):
            # Ensure contiguous bytes for robust behavior across Python versions.
            data = plaintext.tobytes()
        else:
            # Match reference expectation: input must be bytes-like.
            raise TypeError("plaintext must be a bytes-like object")

        digest = hashlib.sha256(data).digest()
        return {"digest": digest}