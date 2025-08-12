import sympy

class Solver:
    def solve(self, problem):
        """
        Solve semiprime factorization using sympy.factorint.
        Returns dict with p < q.
        """
        # Extract composite value
        try:
            composite = sympy.Integer(problem["composite"])
        except Exception as e:
            raise ValueError(f"Invalid composite input: {e}")

        # Factor using sympy
        fdict = sympy.factorint(composite)
        # Expand factors by exponent
        factors = [prime for prime, exp in fdict.items() for _ in range(exp)]

        # Expect exactly two factors
        if len(factors) != 2:
            raise ValueError(f"Expected 2 prime factors, got {factors}")

        # Sort and return
        p, q = sorted(factors)
        return {"p": int(p), "q": int(q)}