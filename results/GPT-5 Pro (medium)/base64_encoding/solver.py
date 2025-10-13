from typing import Any, Dict
from binascii import b2a_base64

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, bytes]:
        """
        Encode the provided plaintext bytes using standard Base64.

        Args:
            problem (dict): Must contain key 'plaintext' with bytes-like object.

        Returns:
            dict: {'encoded_data': Base64-encoded bytes}
        """
        plaintext = problem["plaintext"]
        return {"encoded_data": b2a_base64(plaintext, newline=False)}