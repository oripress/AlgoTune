import base64
import hmac
from typing import Any, Union
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        plaintext = problem["plaintext"]
        
        try:
            # Use the built-in base64 module for maximum speed
            encoded_data = base64.b64encode(plaintext)
            return {"encoded_data": encoded_data}
        except Exception as e:
            # Fallback implementation if base64 fails
            base64_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
            
            # Handle empty input
            if len(plaintext) == 0:
                return {"encoded_data": b""}
            
            # Convert to bytes for faster access
            data = plaintext
            length = len(data)
            
            # Process 3 bytes at a time
            result = ""
            i = 0
            while i < length - 2:
                # Get 3 bytes
                b1 = data[i]
                b2 = data[i + 1]
                b3 = data[i + 2]
                
                # Convert to 4 base64 characters
                result += base64_chars[(b1 >> 2) & 0x3F]
                result += base64_chars[((b1 & 0x03) << 4) | ((b2 >> 4) & 0x0F)]
                result += base64_chars[((b2 & 0x0F) << 2) | ((b3 >> 6) & 0x03)]
                result += base64_chars[b3 & 0x3F]
                
                i += 3
            
            # Handle remaining bytes
            remaining = length - i
            if remaining == 1:
                b1 = data[i]
                result += base64_chars[(b1 >> 2) & 0x3F]
                result += base64_chars[(b1 & 0x03) << 4]
                result += "=="
            elif remaining == 2:
                b1 = data[i]
                b2 = data[i + 1]
                result += base64_chars[(b1 >> 2) & 0x3F]
                result += base64_chars[((b1 & 0x03) << 4) | ((b2 >> 4) & 0x0F)]
                result += base64_chars[(b2 & 0x0F) << 2]
                result += "="
            
            return {"encoded_data": result.encode('ascii')}