import hashlib

class Solver:
    def solve(self, problem, **kwargs):
        plaintext = problem["plaintext"]
        return {"digest": hashlib.sha256(plaintext).digest()}