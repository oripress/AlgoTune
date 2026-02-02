import numpy as np
import scipy.ndimage

class Solver:
    def solve(self, problem, **kwargs):
        image = np.array(problem["image"], dtype=float)
        matrix = np.array(problem["matrix"], dtype=float)
        
        transformed_image = scipy.ndimage.affine_transform(
            image, matrix, order=3, mode='constant', cval=0.0
        )
        
        return {"transformed_image": transformed_image.tolist()}