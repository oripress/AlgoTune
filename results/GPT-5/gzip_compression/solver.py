from typing import Any
import zlib

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, bytes]:
        plaintext = problem["plaintext"]
        compressor = zlib.compressobj(level=9, wbits=16 + zlib.MAX_WBITS)  # gzip wrapper
        compressed_data = compressor.compress(plaintext) + compressor.flush()
        return {"compressed_data": compressed_data}