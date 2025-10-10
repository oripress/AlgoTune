import numpy as np

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> np.ndarray:
        """
        Optimized matrix multiplication using NumPy's @ operator.
        
        Args:
            problem (dict): A dictionary with keys "A" and "B".
        
        Returns:
            np.ndarray: The product matrix C = A @ B.
        """
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        return (A @ B).tolist()