import numpy as np
import ot
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the EMD problem using the POT library's optimized solver.
        
        :param problem: A dictionary representing the EMD problem.
        :return: A dictionary with key "transport_plan" containing the optimal
                 transport plan matrix G as a numpy array.
        """
        # Extract problem data
        a = np.asarray(problem["source_weights"], dtype=np.float64)
        b = np.asarray(problem["target_weights"], dtype=np.float64)
        M = np.asarray(problem["cost_matrix"], dtype=np.float64)
        
        # Compute the optimal transport plan using the EMD algorithm
        # This is more efficient than general linear programming
        G = ot.emd(a, b, M)
        
        return {"transport_plan": G}