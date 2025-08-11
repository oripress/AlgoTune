try:
    import _base64 as _b64mod
    _b64 = _b64mod.b64encode
except ImportError:
    from base64 import b64encode as _b64

class Solver:
    def solve(self, problem, **kwargs) -> dict[str, bytes]:
        # Direct C‐extension call to Base64 encoder
        return {"encoded_data": _b64(problem["plaintext"])}