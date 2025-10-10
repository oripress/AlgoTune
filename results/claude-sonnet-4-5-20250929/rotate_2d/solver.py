import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        self.reshape = False
        self.order = 3
        self.mode = 'constant'
    
    def solve(self, problem):
        """
        Optimized solver - minimize overhead.
        """
        image = problem["image"]
        angle = problem["angle"]
        
        # Direct rotation without unnecessary conversions
        rotated_image = scipy.ndimage.rotate(
            image, angle, reshape=self.reshape, order=self.order, mode=self.mode
        )
        
        # Return as list
        return {"rotated_image": rotated_image.tolist()}