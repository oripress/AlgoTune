import sympy

class Solver:
    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        """
        Solve the integer factorization problem by finding the prime factors of the composite number.
        Optimized for the case where the composite is a product of exactly two primes.
        """
        composite_val = problem["composite"]
        
        # Handle the case where composite might already be an int
        if isinstance(composite_val, int):
            composite = composite_val
        else:
            # Convert to int for large numbers
            composite = int(composite_val)
        
        # Use sympy's factorization - convert to sympy Integer for large number handling
        factors_dict = sympy.factorint(composite)
        
        # Extract the prime factors, accounting for exponents
        primes = []
        for prime, exp in factors_dict.items():
            for _ in range(exp):
                primes.append(int(prime))  # Convert sympy Integer to regular int
        
        # Should always have exactly 2 factors for this problem (could be the same prime twice)
        if len(primes) != 2:
            # This shouldn't happen for valid inputs, but let's handle it gracefully
            # If there's only one prime with exponent 2, duplicate it
            if len(factors_dict) == 1 and sum(factors_dict.values()) == 2:
                prime = list(factors_dict.keys())[0]
                primes = [int(prime), int(prime)]
            else:
                raise ValueError(f"Expected 2 factors, but got {len(primes)}")
        
        # Sort to ensure p <= q (not strictly <, as they could be equal if it's p^2)
        p, q = sorted(primes)
        
        return {"p": p, "q": q}