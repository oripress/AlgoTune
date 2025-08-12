import hashlib
from typing import Any, Dict

class Solver:
    def __init__(self):
        self.hasher = hashlib.sha256()
    
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, bytes]:
        self.hasher.update(problem["plaintext"])
        digest = self.hasher.digest()
        self.hasher = hashlib.sha256()  # Create new instance instead of reset
        return {"digest": digest}