import zlib
from typing import Any
try:
    from fast_gzip import fast_compress
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Compress the plaintext using optimized gzip compression.
        
        Args:
            problem (dict): The problem dictionary containing 'plaintext'.
            
        Returns:
            dict: A dictionary containing 'compressed_data'.
        """
        plaintext = problem["plaintext"]
        
        if USE_CYTHON:
            compressed_data = fast_compress(plaintext)
        else:
            # Fallback to zlib
            compressor = zlib.compressobj(level=9, wbits=31)
            compressed_data = compressor.compress(plaintext)
            compressed_data += compressor.flush()
        
        return {"compressed_data": compressed_data}