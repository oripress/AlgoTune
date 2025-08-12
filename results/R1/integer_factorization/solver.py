import sympy

class Solver:
    def solve(self, problem, **kwargs):
        composite_val = problem["composite"]
        factors = sympy.factorint(composite_val, multiple=True)
        if len(factors) != 2:
            raise ValueError(f"Expected exactly two prime factors, but found {len(factors)} factors")
        p, q = sorted(factors)
        return {"p": int(p), "q": int(q)}