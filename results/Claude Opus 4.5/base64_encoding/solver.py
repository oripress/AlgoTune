import base64

try:
    from base64_fast import base64_encode_fast
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

class Solver:
    def solve(self, problem, **kwargs):
        plaintext = problem["plaintext"]
        n = len(plaintext)
        
        # Use Cython for larger inputs
        if HAS_CYTHON and n > 1000:
            return {"encoded_data": base64_encode_fast(plaintext)}
        
        return {"encoded_data": base64.b64encode(plaintext)}