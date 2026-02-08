import binascii
from typing import Any

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        plaintext = problem["plaintext"]
        encoded_data = binascii.b2a_base64(plaintext, newline=False)
        return {"encoded_data": encoded_data}