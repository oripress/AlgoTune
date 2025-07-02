import logging
import random

import sympy

from AlgoTuneTasks.base import register_task, Task


@register_task("integer_factorization")
class IntegerFactorization(Task):
    def __init__(self, **kwargs):
        """
        Initialize the IntegerFactorization task.

        In this task, you are given a composite number that is a product of two large prime numbers.
        The task is to find these two prime factors.
        """
        super().__init__(**kwargs)

    def generate_problem(self, n: int, random_seed: int = 1) -> dict[str, int]:
        """
        Generate a composite number that is a product of two prime numbers.

        :param n: Parameter controlling the size of the problem. Each prime will be 8*max(1,n) bits long.
        :param random_seed: Seed for reproducibility.
        :return: A dictionary with key "composite" representing the product of two n-bit primes.
        """
        logging.debug(
            f"Generating integer factorization problem with n={n} and random_seed={random_seed}"
        )
        rng = random.Random(random_seed)

        # Calculate the actual bit length for each prime
        n_bits = 8 * max(1, n)

        # Generate two different primes with the calculated bit length
        p = sympy.nextprime(rng.randint(2 ** (n_bits - 1), 2**n_bits - 1))
        q = sympy.nextprime(p)

        # Compute the composite number
        composite = p * q

        logging.debug(
            f"Generated integer factorization problem with composite={composite} "
            f"(primes of {n_bits} bits each)"
        )
        return {"composite": composite}

    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        """
        Solve the integer factorization problem by finding the prime factors of the composite number.

        For a proper solution, one would need to implement a factorization algorithm like:
        - Trial division
        - Pollard's rho algorithm
        - Quadratic sieve
        - General number field sieve

        In this reference implementation, we use sympy's factorization capabilities.

        :param problem: A dictionary containing the composite number.
        :return: A dictionary with keys "p" and "q" containing the two prime factors, where p < q.
        :raises ValueError: If the factorization does not result in exactly two prime factors.
        """
        composite_val = problem["composite"]

        # Ensure composite_val is a SymPy Integer before passing to factorint
        try:
            composite = sympy.Integer(composite_val)
        except (TypeError, ValueError) as e:
            raise ValueError(f"The composite value '{composite_val}' could not be converted to a SymPy Integer: {e}")

        # Extract the prime factors using sympy's factorization
        factors = [prime for prime, exp in sympy.factorint(composite).items() for _ in range(exp)]

        # Ensure we have exactly two factors (should always be the case for our generated problems)
        if len(factors) != 2:
            raise ValueError(f"Expected 2 factors, but got {len(factors)}.")

        # Sort the factors to ensure p < q
        p, q = sorted(factors)

        return {"p": p, "q": q}

    def is_solution(self, problem: dict[str, int], solution: dict[str, int]) -> bool:
        """
        Check if the factorization solution is valid.

        This method checks:
          - The solution is a dictionary.
          - The solution contains the 'p' and 'q' keys.
          - Both p and q are prime numbers.
          - p < q.
          - The product p*q equals the original composite number.

        :param problem: A dictionary containing the problem with key "composite".
        :param solution: A dictionary containing the factors with keys "p" and "q".
        :return: True if the solution is valid, False otherwise.
        """
        composite = problem.get("composite")
        if composite is None:
            logging.error("Problem does not contain 'composite'.")
            return False

        # Check that solution is a dict first
        if not isinstance(solution, dict):
            logging.error("Solution is not a dictionary.")
            return False

        if "p" not in solution or "q" not in solution:
            logging.error("Solution does not contain 'p' and 'q' keys.")
            return False

        p = solution["p"]
        q = solution["q"]

        # Check that both factors are integers
        if not isinstance(p, int) or not isinstance(q, int):
            logging.error("Factors must be integers.")
            return False

        # Check that both factors are prime using sympy
        if not sympy.isprime(p):
            logging.error(f"Factor {p} is not prime.")
            return False

        if not sympy.isprime(q):
            logging.error(f"Factor {q} is not prime.")
            return False

        # Check that p < q
        if p >= q:
            logging.error(f"The factors must be ordered such that p < q, but got p={p}, q={q}.")
            return False

        # Check that the product of the factors equals the composite number
        if p * q != composite:
            logging.error(
                f"Product of p*q ({p}*{q}={p * q}) does not equal composite number ({composite})."
            )
            return False

        # All checks passed
        return True
