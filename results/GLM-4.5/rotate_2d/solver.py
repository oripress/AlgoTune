import numpy as np
import scipy.ndimage
from scipy.ndimage import rotate

class Solver:
    def __init__(self):
        pass
    
    def solve(self, problem, **kwargs):
        # Directly access the values and convert to numpy array efficiently
        image = np.asarray(problem["image"], dtype=np.float64)
        angle = problem["angle"]
        
        try:
            # Use a more direct approach
            rotated_image = rotate(image, angle, reshape=False, order=3, mode='constant', cval=0.0)
        except Exception as e:
            return {"rotated_image": []}  # Indicate failure

        # Convert to list using the most efficient approach
        rotated_list = rotated_image.tolist()
        
        return {"rotated_image": rotated_list}