import binascii

_b2a_base64 = binascii.b2a_base64

class Solver:
    __slots__ = ()

    def solve(self, problem, _b2a_base64=_b2a_base64, **kwargs):
        return {"encoded_data": _b2a_base64(problem["plaintext"], newline=False)}