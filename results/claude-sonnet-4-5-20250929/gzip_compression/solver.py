import gzip
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Compress the plaintext using gzip.
        
        Args:
            problem (dict): The problem dictionary containing 'plaintext'.
        
        Returns:
            dict: A dictionary containing 'compressed_data'.
        """
        return {"compressed_data": gzip.compress(problem["plaintext"], compresslevel=9, mtime=0)}