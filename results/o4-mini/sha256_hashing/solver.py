import hashlib
_sha256 = hashlib.sha256

class Solver:
    def solve(self, problem: dict, **kwargs):
        data = problem["plaintext"]
        # one-shot C-optimized constructor with data, then finalize
        return {"digest": _sha256(data).digest()}