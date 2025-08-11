# Try using scipy for potential optimizations
import base64
import scipy

class Solver:
    def solve(self, problem, **kwargs):
        # Use the standard library implementation which achieved 1.00x speedup
        return {"encoded_data": base64.b64encode(problem["plaintext"])}