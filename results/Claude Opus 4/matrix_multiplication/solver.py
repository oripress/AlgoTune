import numpy as np

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[list[float]]:
        """
        Solve the matrix multiplication task by computing C = A · B.
        
        Args:
            problem (dict): A dictionary with keys "A" and "B".
        
        Returns:
            list: A list of lists of numbers representing the product matrix C.
        """
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        C = np.dot(A, B)
        return C.tolist()