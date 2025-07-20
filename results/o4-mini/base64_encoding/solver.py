import binascii
_b2a = binascii.b2a_base64

class Solver:
    def solve(self, problem, **kwargs):
        # Retrieve plaintext and materialize if needed
        data = problem["plaintext"]
        if not isinstance(data, (bytes, bytearray)):
            try:
                data = bytes(data)
            except Exception:
                try:
                    data = data.tobytes()
                except Exception:
                    pass
        # Fast C-level Base64 encode without newline
        encoded = _b2a(data, newline=False)
        return {"encoded_data": encoded}