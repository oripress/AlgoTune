import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        self.order = 3  # cubic spline interpolation
        self.mode = 'constant'  # boundary handling with 0 padding
    
    def solve(self, problem, **kwargs):
        """
        Apply 2D affine transformation to an input image.
        
        :param problem: Dictionary with 'image' and 'matrix' keys
        :return: Dictionary with 'transformed_image' key
        """
        image = problem["image"]
        matrix = problem["matrix"]
        
        # Perform affine transformation
        transformed_image = scipy.ndimage.affine_transform(
            image, matrix, order=self.order, mode=self.mode
        )
        
        return {"transformed_image": transformed_image.tolist()}