from brent_cython import brent

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        composite = int(problem["composite"])
        p = brent(composite)
        q = composite // p
        if p > q:
            p, q = q, p
        return {"p": p, "q": q}