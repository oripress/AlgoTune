import zlib
import struct
from concurrent.futures import ThreadPoolExecutor

class Solver:
    def __init__(self):
        self.header = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff'
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def solve(self, problem, **kwargs):
        plaintext = problem["plaintext"]
        
        # Submit CRC calculation to thread pool (runs in parallel since zlib releases GIL)
        crc_future = self.executor.submit(zlib.crc32, plaintext)
        
        # Compress in main thread
        co = zlib.compressobj(9, zlib.DEFLATED, -15)
        compressed = co.compress(plaintext) + co.flush()
        
        # Get CRC result
        crc = crc_future.result() & 0xffffffff
        size = len(plaintext) & 0xffffffff
        
        return {"compressed_data": self.header + compressed + struct.pack('<II', crc, size)}