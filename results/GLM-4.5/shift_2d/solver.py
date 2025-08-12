import numpy as np
from typing import Any
import scipy.ndimage

class Solver:
    def __init__(self):
        self.order = 3  # cubic spline interpolation
        self.mode = 'constant'  # padding with 0
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the 2D shift problem using scipy.ndimage.shift.
        
        :param problem: A dictionary representing the problem.
        :return: A dictionary with key "shifted_image":
                 "shifted_image": The shifted image as a list of lists.
        """
        image = problem["image"]
        shift_vector = problem["shift"]
        
        try:
            # Convert input to numpy array
            image_array = np.asarray(image, dtype=np.float64)
            
            # Use scipy's shift function
            shifted_image = scipy.ndimage.shift(
                image_array, shift_vector, order=self.order, mode=self.mode
            )
            
        except Exception as e:
            return {"shifted_image": []}  # Indicate failure
        
        # Convert back to list of lists as expected by validation
        solution = {"shifted_image": shifted_image.tolist()}
        return solution