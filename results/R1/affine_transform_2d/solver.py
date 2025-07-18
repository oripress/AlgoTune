import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        # Set the parameters as specified in the problem
        self.order = 3
        self.mode = 'constant'
        
    def solve(self, problem, **kwargs):
        """
        Reference implementation from the problem description
        """
        image = problem["image"]
        matrix = problem["matrix"]
        
        # Perform affine transformation
        try:
            # output_shape can be specified, default is same as input
            transformed_image = scipy.ndimage.affine_transform(
                image, matrix, order=self.order, mode=self.mode
            )
        except Exception as e:
            # Return an empty list to indicate failure
            return {"transformed_image": []}
        
        return {"transformed_image": transformed_image.tolist()}