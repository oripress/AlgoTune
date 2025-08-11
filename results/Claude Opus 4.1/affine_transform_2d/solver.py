import numpy as np
import scipy.ndimage
from typing import Any

class Solver:
    def __init__(self):
        """Initialize the solver with transformation parameters."""
        self.order = 3  # Cubic spline interpolation
        self.mode = 'constant'  # Padding with 0
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Solves the 2D affine transformation problem using scipy.ndimage.affine_transform.
        
        :param problem: A dictionary representing the problem.
        :return: A dictionary with key "transformed_image".
        """
        image = np.asarray(problem["image"], dtype=np.float64)
        matrix = np.asarray(problem["matrix"], dtype=np.float64)
        
        # Perform affine transformation
        try:
            transformed_image = scipy.ndimage.affine_transform(
                image, matrix, order=self.order, mode=self.mode
            )
        except Exception:
            return {"transformed_image": []}
        
        # Convert back to list of lists for output
        solution = {"transformed_image": transformed_image.tolist()}
        return solution