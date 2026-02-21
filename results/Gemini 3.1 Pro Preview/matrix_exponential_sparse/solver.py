import ast
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm

class Solver:
    def solve(self, problem: dict[str, sparse.spmatrix]) -> sparse.spmatrix:
        """
        Solve the sparse matrix exponential problem by computing exp(A).
        Uses scipy.sparse.linalg.expm to compute the matrix exponential.

        :param problem: A dictionary representing the matrix exponential problem.
        :return: The matrix exponential of the input matrix A, represented as sparse matrix.

        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        A = problem["matrix"]
        solution = expm(A)

        return solution