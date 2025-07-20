import base64

class Solver:
    def solve(self, problem, **kwargs):
        """Optimized Base64 encoding using the standard library."""
        plaintext = problem["plaintext"]
        encoded_data = base64.b64encode(plaintext)
        return {"encoded_data": encoded_data}