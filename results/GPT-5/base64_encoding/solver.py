from __future__ import annotations

import binascii
from typing import Any, Dict

# Bind at module load to avoid repeated attribute lookups during solve calls
_B2A64 = binascii.b2a_base64

class Solver:
    __slots__ = ()

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, bytes]:
        """
        Encode the provided plaintext bytes using standard Base64.

        Args:
            problem: Dictionary with key 'plaintext' containing bytes to encode.

        Returns:
            Dictionary with key 'encoded_data' containing Base64-encoded bytes.
        """
        plaintext = problem["plaintext"]
        return {"encoded_data": _B2A64(plaintext, newline=False)}