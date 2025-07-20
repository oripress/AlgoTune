from typing import Any
import numba
import numpy as np

@numba.jit(nopython=True, cache=True)
def _encode_base64_numba_impl(data: np.ndarray, table: np.ndarray) -> bytes:
    """Fast Base64 encoding using Numba JIT compilation."""
    n = len(data)
    
    # Calculate output size
    output_size = ((n + 2) // 3) * 4
    result = np.empty(output_size, dtype=np.uint8)
    
    # Process in chunks of 3 bytes
    i = 0
    j = 0
    while i < n - 2:
        # Get 3 bytes
        b1 = data[i]
        b2 = data[i+1]
        b3 = data[i+2]
        
        # Extract 4 6-bit values
        result[j] = table[b1 >> 2]
        result[j+1] = table[((b1 & 0x03) << 4) | (b2 >> 4)]
        result[j+2] = table[((b2 & 0x0F) << 2) | (b3 >> 6)]
        result[j+3] = table[b3 & 0x3F]
        
        i += 3
        j += 4
    
    # Handle remaining bytes
    if i < n:
        b1 = data[i]
        result[j] = table[b1 >> 2]
        
        if i + 1 < n:
            b2 = data[i+1]
            result[j+1] = table[((b1 & 0x03) << 4) | (b2 >> 4)]
            result[j+2] = table[(b2 & 0x0F) << 2]
            result[j+3] = 61  # ord('=')
        else:
            result[j+1] = table[(b1 & 0x03) << 4]
            result[j+2] = 61  # ord('=')
            result[j+3] = 61  # ord('=')
    
    return result.tobytes()

class Solver:
    def __init__(self):
        # Pre-compute the Base64 encoding table
        self.b64_table = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        self.b64_array = np.frombuffer(self.b64_table, dtype=np.uint8)
        
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """Encode the plaintext using optimized Base64 algorithm."""
        plaintext = problem["plaintext"]
        
        # Materialize memory-mapped or array data into raw bytes
        if not isinstance(plaintext, (bytes, bytearray)):
            try:
                plaintext = bytes(plaintext)
            except Exception:
                try:
                    plaintext = plaintext.tobytes()
                except Exception:
                    pass
        
        # Convert to numpy array for numba
        data = np.frombuffer(plaintext, dtype=np.uint8)
        encoded = _encode_base64_numba_impl(data, self.b64_array)
        
        return {"encoded_data": encoded}