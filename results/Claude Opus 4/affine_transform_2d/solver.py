import numpy as np
from typing import Any

class Solver:
    def __init__(self):
        self.order = 3  # Cubic spline interpolation
        self.mode = 'constant'  # Padding with 0
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the 2D affine transformation problem.
        
        :param problem: A dictionary representing the problem.
        :return: A dictionary with key "transformed_image"
        """
        image = np.array(problem["image"], dtype=np.float64)
        matrix = np.array(problem["matrix"], dtype=np.float64)
        
        # Use scipy for now to ensure correctness
        import scipy.ndimage
        transformed_image = scipy.ndimage.affine_transform(
            image, matrix, order=self.order, mode=self.mode
        )
        
        return {"transformed_image": transformed_image.tolist()}