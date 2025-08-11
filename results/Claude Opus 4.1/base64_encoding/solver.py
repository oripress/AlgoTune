import numpy as np
from typing import Any
import numba as nb

# Pre-compile the base64 encoding function with Numba
@nb.njit(cache=True, fastmath=True)
def encode_base64_numba(data, b64_table):
    data_len = len(data)
    
    # Calculate output size
    remainder = data_len % 3
    padding_len = (3 - remainder) if remainder else 0
    num_groups = (data_len + 2) // 3
    output_len = num_groups * 4
    
    # Allocate output array
    output = np.empty(output_len, dtype=np.uint8)
    
    # Process full 3-byte groups
    full_groups = data_len // 3
    for i in range(full_groups):
        idx = i * 3
        out_idx = i * 4
        
        # Extract 3 bytes
        b0 = data[idx]
        b1 = data[idx + 1]
        b2 = data[idx + 2]
        
        # Convert to 4 base64 characters
        output[out_idx] = b64_table[(b0 >> 2) & 0x3F]
        output[out_idx + 1] = b64_table[((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F)]
        output[out_idx + 2] = b64_table[((b1 & 0x0F) << 2) | ((b2 >> 6) & 0x03)]
        output[out_idx + 3] = b64_table[b2 & 0x3F]
    
    # Handle remaining bytes
    if remainder == 1:
        idx = full_groups * 3
        out_idx = full_groups * 4
        b0 = data[idx]
        
        output[out_idx] = b64_table[(b0 >> 2) & 0x3F]
        output[out_idx + 1] = b64_table[(b0 & 0x03) << 4]
        output[out_idx + 2] = ord('=')
        output[out_idx + 3] = ord('=')
    elif remainder == 2:
        idx = full_groups * 3
        out_idx = full_groups * 4
        b0 = data[idx]
        b1 = data[idx + 1]
        
        output[out_idx] = b64_table[(b0 >> 2) & 0x3F]
        output[out_idx + 1] = b64_table[((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F)]
        output[out_idx + 2] = b64_table[(b1 & 0x0F) << 2]
        output[out_idx + 3] = ord('=')
    
    return output

class Solver:
    def __init__(self):
        # Create Base64 encoding table as numpy array for Numba
        b64_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
        self.b64_table = np.array([ord(c) for c in b64_chars], dtype=np.uint8)
    
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Encode the plaintext using optimized Base64 algorithm with Numba.
        
        Args:
            problem (dict): The problem dictionary with 'plaintext' key.
        
        Returns:
            dict: A dictionary containing 'encoded_data'.
        """
        plaintext = problem["plaintext"]
        
        # Convert to numpy array
        data = np.frombuffer(plaintext, dtype=np.uint8)
        
        # Use Numba-compiled function
        encoded_array = encode_base64_numba(data, self.b64_table)
        
        return {"encoded_data": bytes(encoded_array)}