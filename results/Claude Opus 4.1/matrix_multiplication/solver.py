import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the matrix multiplication task by computing C = A · B.
        
        Args:
            problem (dict): A dictionary with keys "A" and "B".
            
        Returns:
            list: The product matrix C as a list of lists.
        """
        # Use asarray with order='C' for contiguous memory layout
        A = np.asarray(problem["A"], dtype=np.float64, order='C')
        B = np.asarray(problem["B"], dtype=np.float64, order='C')
        
        # Use @ operator which may be slightly faster than np.dot
        C = A @ B
        
        return C.tolist()