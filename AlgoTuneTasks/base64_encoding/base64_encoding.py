import base64
import hmac
import logging

# Removed math, string imports
from typing import Any, Union  # Standard library

import numpy as np  # Third-party needed for seeded random bytes

from AlgoTuneTasks.base import register_task, Task  # Local application


@register_task("base64_encoding")
class Base64Encoding(Task):
    """
    Base64Encoding Task:

    Encode binary data (generated using a Zipfian distribution of words)
    using the standard Base64 algorithm.
    Encode binary data using the standard Base64 algorithm.
    The difficulty scales with the size of the input data.
    """

    # Constants for data generation
    DEFAULT_PLAINTEXT_MULTIPLIER = 2048  # Target bytes of plaintext per unit of n

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_problem(self, n: int, random_seed: int = 1) -> dict[str, Any]:
        """
        Generate random binary data for the Base64 encoding task.

        Args:
            n (int): Scaling parameter, determines the target plaintext size.
            random_seed (int): Seed for reproducibility.

        Returns:
            dict: A dictionary containing the problem parameters (plaintext).
        """
        # Use n to scale the target plaintext size
        target_plaintext_size = max(0, n * self.DEFAULT_PLAINTEXT_MULTIPLIER)

        logging.debug(
            f"Generating Base64 encoding problem (random bytes) with n={n} "
            f"(target_plaintext_size={target_plaintext_size}), random_seed={random_seed}"
        )

        if target_plaintext_size == 0:
            logging.debug("Target size is 0, returning empty bytes.")
            return {"plaintext": b""}

        # Seed the numpy random number generator for reproducibility
        rng = np.random.default_rng(random_seed)

        # Generate random bytes using the seeded generator
        plaintext_bytes = rng.bytes(target_plaintext_size)

        logging.debug(f"Generated final plaintext size: {len(plaintext_bytes)} bytes")
        return {"plaintext": plaintext_bytes}

    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Encode the plaintext using the Base64 algorithm.

        Args:
            problem (dict): The problem dictionary generated by `generate_problem`.

        Returns:
            dict: A dictionary containing 'encoded_data'.
        """
        plaintext = problem["plaintext"]

        try:
            # Encode the data using standard Base64
            encoded_data = base64.b64encode(plaintext)
            return {"encoded_data": encoded_data}

        except Exception as e:
            logging.error(f"Error during Base64 encoding in solve: {e}")
            raise  # Re-raise exception

    def is_solution(self, problem: dict[str, Any], solution: Union[dict[str, bytes], Any]) -> bool:
        """
        Verify the provided solution by comparing its encoded data
        against the result obtained from calling the task's own solve() method.

        Args:
            problem (dict): The problem dictionary.
            solution (dict): The proposed solution dictionary with 'encoded_data'.

        Returns:
            bool: True if the solution matches the result from self.solve().
        """
        if not isinstance(solution, dict) or "encoded_data" not in solution:
            logging.error(
                f"Invalid solution format. Expected dict with 'encoded_data'. Got: {type(solution)}"
            )
            return False

        try:
            # Get the correct result by calling the solve method
            reference_result = self.solve(problem)
            reference_encoded_data = reference_result["encoded_data"]
        except Exception as e:
            # If solve itself fails, we cannot verify the solution
            logging.error(f"Failed to generate reference solution in is_solution: {e}")
            return False

        solution_encoded_data = solution["encoded_data"]

        # Ensure type is bytes before comparison
        if not isinstance(solution_encoded_data, bytes):
            logging.error("Solution 'encoded_data' is not bytes.")
            return False

        # Direct comparison is sufficient for Base64 output.
        # Using hmac.compare_digest for consistency and potential timing attack resistance.
        encoded_data_match = hmac.compare_digest(reference_encoded_data, solution_encoded_data)

        return encoded_data_match
