import zlib
import concurrent.futures
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, bytes]:
        plaintext = problem["plaintext"]
        n = len(plaintext)
        
        if n < 500000:
            return {"compressed_data": zlib.compress(plaintext, 9, 31)}
            
        num_chunks = 2
        chunk_size = (n + num_chunks - 1) // num_chunks
        chunks = [plaintext[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
            compressed_chunks = list(executor.map(lambda c: zlib.compress(c, 9, 31), chunks))
            
        return {"compressed_data": b"".join(compressed_chunks)}