import numpy as np
from numba import njit
from typing import Any

# Base64 alphabet
BASE64_ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

@njit
def base64_encode_numba(data, alphabet):
    """Numba-optimized Base64 encoding."""
    n = len(data)
    # Calculate output length: each 3 bytes -> 4 base64 chars
    output_len = ((n + 2) // 3) * 4
    result = np.zeros(output_len, dtype=np.uint8)
    
    i = 0
    j = 0
    while i < n:
        # Get 3 bytes (or less for the last group)
        b1 = data[i]
        b2 = data[i + 1] if i + 1 < n else 0
        b3 = data[i + 2] if i + 2 < n else 0
        
        # Convert to 4 base64 characters
        result[j] = alphabet[(b1 >> 2) & 0x3F]
        result[j + 1] = alphabet[((b1 & 0x03) << 4) | ((b2 >> 4) & 0x0F)]
        
        if i + 1 < n:
            result[j + 2] = alphabet[((b2 & 0x0F) << 2) | ((b3 >> 6) & 0x03)]
        else:
            result[j + 2] = ord('=')
            
        if i + 2 < n:
            result[j + 3] = alphabet[b3 & 0x3F]
        else:
            result[j + 3] = ord('=')
        
        i += 3
        j += 4
    
    return result

class Solver:
    def __init__(self):
        # Pre-compute alphabet as numpy array for Numba
        self.alphabet = np.frombuffer(BASE64_ALPHABET, dtype=np.uint8)
    
    def solve(self, problem, **kwargs) -> dict:
        """Encode plaintext using optimized Base64 encoding."""
        plaintext = problem["plaintext"]
        
        # Convert to numpy array for Numba processing
        data = np.frombuffer(plaintext, dtype=np.uint8)
        
        # Encode using Numba
        encoded_array = base64_encode_numba(data, self.alphabet)
        
        # Convert back to bytes
        encoded_data = encoded_array.tobytes()
        
        return {"encoded_data": encoded_data}