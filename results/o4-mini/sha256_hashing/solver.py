import hashlib
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, bytes]:
        data = problem["plaintext"]
        # One-shot C-optimized SHA-256
        digest = hashlib.sha256(data).digest()
        return {"digest": digest}