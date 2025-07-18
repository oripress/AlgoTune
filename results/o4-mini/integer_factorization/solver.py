import sympy

class Solver:
    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        composite_val = problem["composite"]
        try:
            composite = sympy.Integer(composite_val)
        except (TypeError, ValueError) as e:
            raise ValueError(f"The composite value '{composite_val}' could not be converted to a SymPy Integer: {e}")

        # Extract prime factors with multiplicity
        factors = [int(prime) for prime, exp in sympy.factorint(composite).items() for _ in range(exp)]

        if len(factors) != 2:
            raise ValueError(f"Expected 2 factors, but got {len(factors)}.")

        p, q = sorted(factors)
        return {"p": p, "q": q}