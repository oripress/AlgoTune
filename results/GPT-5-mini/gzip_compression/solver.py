import zlib
from typing import Any, Dict

class Solver:
    def __init__(self) -> None:
        # Prebind to reduce per-call attribute lookups
        self._compressobj = zlib.compressobj
        self._level = 9
        # wbits with 16 adds the gzip wrapper (same as 16 + MAX_WBITS)
        self._wbits = zlib.MAX_WBITS | 16
        self._deflated = zlib.DEFLATED

    def solve(self, problem: Any, **kwargs) -> Dict[str, bytes]:
        """
        Compress plaintext to a gzip stream using zlib with the gzip wrapper
        (wbits=16+MAX_WBITS). This lets zlib produce the gzip header/footer in C,
        reducing Python overhead.
        """
        if isinstance(problem, dict):
            plaintext = problem.get("plaintext", b"")
        else:
            plaintext = getattr(problem, "plaintext", problem)

        # Minimal normalization: avoid copying common bytes-like inputs
        if isinstance(plaintext, str):
            data = plaintext.encode()
        else:
            # Accept bytes, bytearray, memoryview directly; fall back to bytes(...) otherwise
            if isinstance(plaintext, (bytes, bytearray, memoryview)):
                data = plaintext
            else:
                try:
                    data = bytes(plaintext)
                except Exception as e:
                    raise TypeError("problem['plaintext'] must be bytes-like or str") from e

        compressor = self._compressobj(self._level, self._deflated, self._wbits)
        compressed = compressor.compress(data)
        compressed += compressor.flush()
        return {"compressed_data": compressed}