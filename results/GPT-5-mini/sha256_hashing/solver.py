from hashlib import sha256
from typing import Any

class Solver:
    def solve(self, problem: Any, **kwargs):
        """
        Compute SHA-256 digest of the provided plaintext.

        Expects a dict with key "plaintext" (like the reference), or raw bytes-like input.
        """
        # Match reference behavior: use problem["plaintext"] when given a mapping.
        if isinstance(problem, dict):
            data = problem["plaintext"]
        else:
            data = problem

        # Normalize common types without unnecessary overhead
        if isinstance(data, memoryview):
            data = data.tobytes()
        elif isinstance(data, str):
            data = data.encode()
        elif not isinstance(data, (bytes, bytearray)):
            data = bytes(data)

        return {"digest": sha256(data).digest()}