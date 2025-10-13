from typing import Any, Dict
import zlib

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, bytes]:
        """
        Compress the plaintext using zlib with a gzip wrapper (wbits=31) and level=9.
        This yields a valid gzip stream and is typically faster than gzip.compress,
        while matching the reference size within tolerance.

        Args:
            problem (dict): A dictionary containing the key 'plaintext' as bytes.

        Returns:
            dict: A dictionary containing 'compressed_data' as bytes.
        """
        plaintext = problem["plaintext"]
        # Create a compressor that outputs gzip format directly (wbits=31)
        comp = zlib.compressobj(
            level=9,
            method=zlib.DEFLATED,
            wbits=31,           # gzip wrapper
            memLevel=8,         # default zlib memLevel for consistency with reference
            strategy=zlib.Z_DEFAULT_STRATEGY,
        )
        out = comp.compress(plaintext)
        out += comp.flush(zlib.Z_FINISH)
        return {"compressed_data": out}