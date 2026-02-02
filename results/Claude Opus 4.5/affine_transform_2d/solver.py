import numpy as np
from scipy.ndimage import affine_transform

class Solver:
    def solve(self, problem, **kwargs):
        # Ensure C-contiguous arrays for better cache performance
        image = np.ascontiguousarray(problem["image"], dtype=np.float64)
        matrix = np.ascontiguousarray(problem["matrix"], dtype=np.float64)
        
        # Pre-allocate output to avoid allocation overhead
        output = np.empty_like(image)
        
        affine_transform(image, matrix, order=3, mode='constant', output=output)
        
        return {"transformed_image": output}