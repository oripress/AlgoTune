import binascii
from typing import Any
import fast_base64

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, bytes]:
        plaintext = problem["plaintext"]
        # Heuristic: for small inputs, binascii is faster due to overhead
        if len(plaintext) < 128 * 1024: # 128KB threshold
             return {"encoded_data": binascii.b2a_base64(plaintext, newline=False)}

        # Use the Cython implementation for larger inputs
        encoded_data = fast_base64.b64encode(plaintext)
        return {"encoded_data": encoded_data}