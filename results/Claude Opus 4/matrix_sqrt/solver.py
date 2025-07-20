import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[complex]]]]:
        """
        Solves the matrix square root problem using scipy.linalg.sqrtm.
        
        Computes the principal matrix square root X such that X @ X = A.
        
        :param problem: A dictionary representing the matrix square root problem.
        :return: A dictionary with key "sqrtm" containing a dictionary with key:
                 "X": A list of lists representing the principal square root matrix X.
                      Contains complex numbers.
        """
        A = problem["matrix"]
        
        # Compute the principal matrix square root
        try:
            X, _ = scipy.linalg.sqrtm(A, disp=False)
        except Exception as e:
            return {"sqrtm": {"X": []}}
        
        # Convert to list format
        solution = {"sqrtm": {"X": X.tolist()}}
        return solution