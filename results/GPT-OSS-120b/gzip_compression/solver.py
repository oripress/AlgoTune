import gzip
from gzip import compress as _gzip_compress
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, bytes]:
        """
        Compress the given plaintext using gzip with maximum compression level.
        The mtime is set to 0 for deterministic output.
        """
        plaintext = problem["plaintext"]
        # Directly use the locally bound function for a tiny speed gain
        compressed_data = _gzip_compress(plaintext, compresslevel=9, mtime=0)
        return {"compressed_data": compressed_data}