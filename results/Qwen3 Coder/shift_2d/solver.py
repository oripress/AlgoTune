import numpy as np
from scipy.ndimage import shift
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        Solves the 2D shift problem using scipy.ndimage.shift with optimizations.
        
        :param problem: A dictionary representing the problem.
        :return: A dictionary with key "shifted_image".
        """
        # Convert input to numpy array with appropriate dtype
        image = np.asarray(problem["image"], dtype=np.float64)
        shift_vector = problem["shift"]
        
        # Apply shift using scipy's optimized implementation
        shifted_image = shift(image, shift_vector, order=3, mode='constant', cval=0.0)
        
        # Convert to list for output
        return {"shifted_image": shifted_image.tolist()}
        # Apply cubic interpolation
        shifted_image = map_coordinates(image, [row_coords, col_coords], order=3, mode='constant', cval=0.0)
        
        # Convert to list for output
        return {"shifted_image": shifted_image.tolist()}