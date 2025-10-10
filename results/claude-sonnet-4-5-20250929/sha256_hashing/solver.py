import hashlib

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute SHA-256 hash using Python's built-in hashlib for optimal performance.
        Using usedforsecurity=False allows for potential optimizations.
        
        :param problem: A dictionary containing the problem with key "plaintext".
        :return: A dictionary with key "digest" containing the SHA-256 hash value.
        """
        plaintext = problem["plaintext"]
        hash_value = hashlib.sha256(plaintext, usedforsecurity=False).digest()
        return {"digest": hash_value}