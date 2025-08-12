import gzip
import io
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        """Compress the plaintext using the gzip algorithm with optimized settings."""
        plaintext = problem["plaintext"]
        
        # Use direct compression for maximum efficiency
        compressed_data = gzip.compress(plaintext, compresslevel=9, mtime=0)
        return {"compressed_data": compressed_data}