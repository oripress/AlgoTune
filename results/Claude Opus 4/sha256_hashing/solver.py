from typing import Any
import hashlib

class Solver:
    def __init__(self):
        # Pre-create a hasher object to avoid repeated instantiation
        self._hasher = hashlib.sha256
    
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Compute the SHA-256 hash of the plaintext.
        
        :param problem: A dictionary containing the problem with key "plaintext".
        :return: A dictionary with key "digest" containing the SHA-256 hash value.
        """
        # Direct access without intermediate variables
        return {"digest": self._hasher(problem["plaintext"]).digest()}