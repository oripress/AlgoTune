import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        """
        Solves the matrix square root problem using scipy.linalg.sqrtm.

        Computes the principal matrix square root X such that X @ X = A.

        :param problem: A dictionary representing the matrix square root problem.
        :return: A dictionary with key "sqrtm" containing a dictionary with key:
                 "X": A list of lists representing the principal square root matrix X.
                      Contains complex numbers.

        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        A = problem["matrix"]

        # Compute the principal matrix square root
        # disp=False prevents warnings/errors from being printed to stdout/stderr
        # but exceptions can still be raised (e.g., if sqrtm fails)
        try:
            X, _ = scipy.linalg.sqrtm(
                A, disp=False
            )  # Unpack the matrix and ignore the error estimate
        except Exception as e:
            # Decide how to handle failure: return empty/error dict, or re-raise?
            # For benchmark purposes, if the reference solver fails, maybe the
            # problem itself is ill-posed for this task. Let's return an empty list
            # or raise an error specific to the benchmark framework if appropriate.
            # Returning an identifiable failure case:
            return {"sqrtm": {"X": []}}  # Indicate failure with empty list

        # Convert complex numbers to a serializable format if needed,
        # though lists of Python complex numbers are generally fine.
        solution = {"sqrtm": {"X": X.tolist()}}
        return solution