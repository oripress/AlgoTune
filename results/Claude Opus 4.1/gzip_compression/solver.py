import zlib
import struct
import time
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Compress the plaintext using zlib directly to create gzip format.
        This avoids some overhead from the gzip module.
        
        Args:
            problem (dict): The problem dictionary containing 'plaintext'.
        
        Returns:
            dict: A dictionary containing 'compressed_data'.
        """
        plaintext = problem["plaintext"]
        
        # Use zlib directly to compress with level 9
        # Create gzip header and trailer manually for speed
        compress_obj = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
        compressed = compress_obj.compress(plaintext) + compress_obj.flush()
        
        # Build gzip format manually
        # Header: magic (2 bytes) + method (1) + flags (1) + mtime (4) + xfl (1) + os (1)
        header = b'\x1f\x8b'  # Magic number
        header += b'\x08'     # Compression method (deflate)
        header += b'\x00'     # Flags
        header += struct.pack('<I', 0)  # mtime = 0 for deterministic output
        header += b'\x02'     # Extra flags (max compression)
        header += b'\xff'     # OS (unknown)
        
        # Calculate CRC32 and size
        crc = zlib.crc32(plaintext) & 0xffffffff
        size = len(plaintext) & 0xffffffff
        
        # Trailer: CRC32 (4 bytes) + uncompressed size (4 bytes)
        trailer = struct.pack('<II', crc, size)
        
        compressed_data = header + compressed + trailer
        
        return {"compressed_data": compressed_data}