import base64
from typing import Any
import numba
import numpy as np
import sys

# Base character mapping, used for LUT generation and remainder handling.
_B64_CHARS_NP = np.array(list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"), dtype=np.uint8)
_PAD_CHAR = ord('=')

# --- Pre-computed 12-bit to uint16 Lookup Table (LUT) ---
# This LUT maps a 12-bit integer to a single uint16 containing the two
# corresponding Base64 characters. This allows writing 2 bytes at once.
_LUT12_U16 = np.empty(4096, dtype=np.uint16)
if sys.byteorder == 'little':
    for i in range(4096):
        _LUT12_U16[i] = _B64_CHARS_NP[(i >> 6) & 0x3F] | (_B64_CHARS_NP[i & 0x3F] << 8)
else: # big-endian
    for i in range(4096):
        _LUT12_U16[i] = (_B64_CHARS_NP[(i >> 6) & 0x3F] << 8) | _B64_CHARS_NP[i & 0x3F]

@numba.njit(cache=True, nogil=True)
def _numba_b64encode_unrolled(data: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated Base64 encoder using a uint16 LUT and loop unrolling.
    The loop is unrolled by a factor of 4 to increase instruction-level parallelism.
    """
    n = len(data)
    if n == 0:
        return np.empty(0, dtype=np.uint8)

    out_len = ((n + 2) // 3) * 4
    encoded = np.empty(out_len, dtype=np.uint8)
    encoded_u16 = encoded.view(np.uint16)
    
    num_chunks = n // 3
    
    # Main unrolled loop processes 4 chunks (12 input bytes) per iteration.
    unroll_factor = 4
    unrolled_iters = num_chunks // unroll_factor
    
    in_idx = 0
    out_idx_u16 = 0
    for _ in range(unrolled_iters):
        # Chunk 1
        chunk1 = (data[in_idx] << 16) | (data[in_idx+1] << 8) | data[in_idx+2]
        encoded_u16[out_idx_u16] = _LUT12_U16[chunk1 >> 12]
        encoded_u16[out_idx_u16 + 1] = _LUT12_U16[chunk1 & 0xFFF]
        
        # Chunk 2
        chunk2 = (data[in_idx+3] << 16) | (data[in_idx+4] << 8) | data[in_idx+5]
        encoded_u16[out_idx_u16 + 2] = _LUT12_U16[chunk2 >> 12]
        encoded_u16[out_idx_u16 + 3] = _LUT12_U16[chunk2 & 0xFFF]

        # Chunk 3
        chunk3 = (data[in_idx+6] << 16) | (data[in_idx+7] << 8) | data[in_idx+8]
        encoded_u16[out_idx_u16 + 4] = _LUT12_U16[chunk3 >> 12]
        encoded_u16[out_idx_u16 + 5] = _LUT12_U16[chunk3 & 0xFFF]

        # Chunk 4
        chunk4 = (data[in_idx+9] << 16) | (data[in_idx+10] << 8) | data[in_idx+11]
        encoded_u16[out_idx_u16 + 6] = _LUT12_U16[chunk4 >> 12]
        encoded_u16[out_idx_u16 + 7] = _LUT12_U16[chunk4 & 0xFFF]

        in_idx += 12
        out_idx_u16 += 8

    # Process remaining chunks (0 to 3) with a simple loop.
    for i in range(unrolled_iters * unroll_factor, num_chunks):
        rem_in_idx = i * 3
        chunk = (data[rem_in_idx] << 16) | (data[rem_in_idx+1] << 8) | data[rem_in_idx+2]
        encoded_u16[out_idx_u16] = _LUT12_U16[chunk >> 12]
        encoded_u16[out_idx_u16 + 1] = _LUT12_U16[chunk & 0xFFF]
        out_idx_u16 += 2

    # Handle padding (final 1 or 2 bytes of original input).
    rem = n % 3
    if rem > 0:
        out_idx_u8 = num_chunks * 4
        rem_idx = num_chunks * 3
        if rem == 1:
            chunk = data[rem_idx] << 16
            encoded[out_idx_u8] = _B64_CHARS_NP[(chunk >> 18) & 0x3F]
            encoded[out_idx_u8 + 1] = _B64_CHARS_NP[(chunk >> 12) & 0x3F]
            encoded[out_idx_u8 + 2] = _PAD_CHAR
            encoded[out_idx_u8 + 3] = _PAD_CHAR
        elif rem == 2:
            chunk = (data[rem_idx] << 16) | (data[rem_idx + 1] << 8)
            encoded[out_idx_u8] = _B64_CHARS_NP[(chunk >> 18) & 0x3F]
            encoded[out_idx_u8 + 1] = _B64_CHARS_NP[(chunk >> 12) & 0x3F]
            encoded[out_idx_u8 + 2] = _B64_CHARS_NP[(chunk >> 6) & 0x3F]
            encoded[out_idx_u8 + 3] = _PAD_CHAR
            
    return encoded

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        plaintext = problem["plaintext"]
        
        if not isinstance(plaintext, (bytes, bytearray)):
            try: plaintext = bytes(plaintext)
            except Exception:
                try: plaintext = plaintext.tobytes()
                except Exception: return {"encoded_data": base64.b64encode(plaintext)}

        if not plaintext:
            return {"encoded_data": b''}

        input_array = np.frombuffer(plaintext, dtype=np.uint8)
        encoded_array = _numba_b64encode_unrolled(input_array)
        return {"encoded_data": encoded_array.tobytes()}