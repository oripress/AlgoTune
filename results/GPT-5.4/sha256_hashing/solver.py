try:
    from sha256_fast import Solver
except Exception:
    from hashlib import sha256 as _sha256

    class Solver:
        __slots__ = ()

        def solve(self, problem, **kwargs):
            return {"digest": _sha256(problem["plaintext"]).digest()}