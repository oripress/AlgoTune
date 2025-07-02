import logging
import random

import numpy as np
from scipy.linalg import lu

from AlgoTuneTasks.base import register_task, Task


@register_task("lu_factorization")
class LUFactorization(Task):
    def __init__(self, **kwargs):
        """
        Initialize the LUFactorization task.

        In this task, you are given a square matrix A.
        The task is to compute its LU factorization such that:
            A = P L U
        where P is a permutation matrix, L is a lower triangular matrix, and U is an upper triangular matrix.
        """
        super().__init__(**kwargs)

    def generate_problem(self, n: int, random_seed: int = 1) -> dict[str, np.ndarray]:
        """
        Generate a random square matrix A of size n x n.

        Although n is provided as a parameter for scaling, the generated problem contains only the matrix.
        The dimension of the matrix can be inferred directly from the matrix itself.

        :param n: The dimension of the square matrix.
        :param random_seed: Seed for reproducibility.
        :return: A dictionary representing the LU factorization problem with key:
                 "matrix": A numpy array of shape (n, n) representing the square matrix A.
        """
        logging.debug(
            f"Generating LU factorization problem with n={n} and random_seed={random_seed}"
        )
        random.seed(random_seed)
        np.random.seed(random_seed)
        A = np.random.randn(n, n)
        problem = {"matrix": A}
        logging.debug("Generated LU factorization problem.")
        return problem

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Solve the LU factorization problem by computing the LU factorization of matrix A.
        Uses scipy.linalg.lu to compute the decomposition:
            A = P L U

        :param problem: A dictionary representing the LU factorization problem.
        :return: A dictionary with key "LU" containing a dictionary with keys:
                 "P": The permutation matrix.
                 "L": The lower triangular matrix.
                 "U": The upper triangular matrix.
        """
        A = problem["matrix"]
        P, L, U = lu(A)
        solution = {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}
        return solution

    def is_solution(
        self, problem: dict[str, np.ndarray], solution: dict[str, dict[str, list[list[float]]]]
    ) -> bool:
        """
        Check if the LU factorization solution is valid and optimal.

        This method checks:
          - The solution contains the 'LU' key with subkeys 'P', 'L', and 'U'.
          - The dimensions of P, L, and U match the dimensions of the input matrix A.
          - None of the matrices contain any infinities or NaNs.
          - The product P @ L @ U reconstructs the original matrix A within a small tolerance,
            acknowledging that the LU factorization is not unique due to permutations.

        :param problem: A dictionary containing the problem, with key "matrix" as the input matrix.
        :param solution: A dictionary containing the LU factorization solution with key "LU"
                         mapping to a dict with keys "P", "L", "U".
        :return: True if the solution is valid and optimal, False otherwise.
        """
        A = problem.get("matrix")
        if A is None:
            logging.error("Problem does not contain 'matrix'.")
            return False

        # Check that the solution contains the 'LU' key.
        if "LU" not in solution:
            logging.error("Solution does not contain 'LU' key.")
            return False

        lu_solution = solution["LU"]

        # Check that 'P', 'L', and 'U' keys are present.
        for key in ["P", "L", "U"]:
            if key not in lu_solution:
                logging.error(f"Solution LU does not contain '{key}' key.")
                return False

        # Convert lists to numpy arrays
        try:
            P = np.array(lu_solution["P"])
            L = np.array(lu_solution["L"])
            U = np.array(lu_solution["U"])
        except Exception as e:
            logging.error(f"Error converting solution lists to numpy arrays: {e}")
            return False

        n = A.shape[0]

        # Check if dimensions match.
        if P.shape != (n, n) or L.shape != (n, n) or U.shape != (n, n):
            logging.error("Dimension mismatch between input matrix and LU factors.")
            return False

        # Check for infinities or NaNs in matrices.
        for mat, name in zip([P, L, U], ["P", "L", "U"]):
            if not np.all(np.isfinite(mat)):
                logging.error(f"Matrix {name} contains non-finite values (inf or NaN).")
                return False

        # Reconstruct A using the factorization.
        A_reconstructed = P @ L @ U

        # Check if A and A_reconstructed are approximately equal.
        if not np.allclose(A, A_reconstructed, atol=1e-6):
            logging.error(
                "Reconstructed matrix does not match the original matrix within tolerance."
            )
            return False

        # All checks passed
        return True
