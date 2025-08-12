import hashlib
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, bytes]:
        """Compute SHA-256 hash with optimized approach."""
        # Try using a different approach that might be more efficient
        plaintext = problem["plaintext"]
        # Use sha256() with data directly and get digest
        return {"digest": hashlib.sha256(plaintext).digest()}