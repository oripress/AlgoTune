import fast_base64

class Solver:
    def solve(self, problem, **kwargs):
        return {"encoded_data": fast_base64.encode(problem["plaintext"])}