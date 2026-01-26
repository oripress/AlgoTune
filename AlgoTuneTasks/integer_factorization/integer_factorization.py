# Copyright (c) 2025 Ori Press and the AlgoTune contributors
# https://github.com/oripress/AlgoTune
import logging
import numbers
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
        :return: A dictionary with key "composite" representing the product of two primes.
        """
        logging.debug(
            f"Generating integer factorization problem with n={n} and random_seed={random_seed}"
        )
        rng = random.Random(random_seed)

        n_bits = 8 * max(1, n)
        low = 2 ** (n_bits - 1)
        high = 2**n_bits - 1

        # Generate two *independent* primes of ~n_bits (avoid "adjacent primes" which make Fermat too easy).
        # Use rng for determinism: sample random ints in range, then nextprime().
        p0 = rng.randint(low, high)
        p = int(sympy.nextprime(p0))

        q0 = rng.randint(low, high)
        q = int(sympy.nextprime(q0))
        while q == p:
            q0 = rng.randint(low, high)
            q = int(sympy.nextprime(q0))

        # Compute composite (Python int is arbitrary precision; no overflow).
        composite = int(p) * int(q)

        logging.debug(
            f"Generated integer factorization problem with composite={composite} "
            f"(primes of ~{n_bits} bits each)"
        )
        return {"composite": composite}

    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        """
        Reference solver using sympy.factorint.

        :param problem: A dictionary containing the composite number.
        :return: A dictionary with keys "p" and "q" containing the two prime factors, where p < q.
        :raises ValueError: If the factorization does not result in exactly two prime factors.
        """
        composite_val = problem["composite"]

        try:
            composite = sympy.Integer(composite_val)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"The composite value '{composite_val}' could not be converted to a SymPy Integer: {e}"
            )

        factors = [prime for prime, exp in sympy.factorint(composite).items() for _ in range(exp)]
        if len(factors) != 2:
            raise ValueError(f"Expected 2 factors, but got {len(factors)}.")

        p, q = sorted(map(int, factors))
        return {"p": p, "q": q}

    def is_solution(self, problem: dict[str, int], solution: dict[str, int]) -> bool:
        """
        Validate factorization:
        - solution is a dict with 'p' and 'q'
        - p*q == composite
        - p and q are prime
        - order is not enforced (we normalize)

        Accepts:
        - int-like: Python int, sympy.Integer, numpy.integer (numbers.Integral)
        - integer strings (with optional +/- and whitespace)
        - integer-valued floats (e.g., 7.0) if exactly integral

        Rejects:
        - arbitrary objects with custom __int__ (avoid validator games)
        """

        def _to_int_like(x, name: str) -> int:
            # Accept exact integer types (Python int, numpy.integer, sympy.Integer, etc.)
            if isinstance(x, numbers.Integral) and not isinstance(x, bool):
                return int(x)

            # Optional: accept integer-valued floats (e.g., 7.0)
            if isinstance(x, float):
                if x.is_integer():
                    return int(x)
                logging.error(f"{name} is a non-integer float: {x}")
                raise TypeError

            # Accept strings that are pure integers (with optional +/- and whitespace)
            if isinstance(x, str):
                s = x.strip()
                if not s:
                    logging.error(f"{name} is an empty string.")
                    raise TypeError
                if s[0] in "+-":
                    sign = -1 if s[0] == "-" else 1
                    s_num = s[1:]
                else:
                    sign = 1
                    s_num = s
                if s_num.isdigit():
                    return sign * int(s_num)
                logging.error(f"{name} is a non-integer string: {x!r}")
                raise TypeError

            logging.error(f"{name} has unsupported type for integer conversion: {type(x)}")
            raise TypeError

        composite = problem.get("composite")
        if composite is None:
            logging.error("Problem does not contain 'composite'.")
            return False

        try:
            composite = _to_int_like(composite, "composite")
        except Exception:
            return False

        if not isinstance(solution, dict):
            logging.error("Solution is not a dictionary.")
            return False
        if "p" not in solution or "q" not in solution:
            logging.error("Solution does not contain 'p' and 'q' keys.")
            return False

        try:
            p = _to_int_like(solution["p"], "p")
            q = _to_int_like(solution["q"], "q")
        except Exception:
            return False

        # Normalize order (accept either p,q or q,p)
        if p > q:
            p, q = q, p

        # Reject non-positive / trivial factors early
        if p <= 1 or q <= 1:
            logging.error(f"Invalid factors: p={p}, q={q}. Factors must be > 1.")
            return False

        # Product must match (Python int multiply is arbitrary-precision; no overflow)
        if p * q != composite:
            logging.error(
                f"Product of p*q ({p}*{q}={p * q}) does not equal composite ({composite})."
            )
            return False

        # Primality checks
        if not sympy.isprime(p):
            logging.error(f"Factor {p} is not prime.")
            return False
        if not sympy.isprime(q):
            logging.error(f"Factor {q} is not prime.")
            return False

        return True
