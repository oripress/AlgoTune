import scipy.ndimage
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        try:
            from PIL import Image
            # print("PIL is available")
        except ImportError:
            # print("PIL is NOT available")
            pass
            
        image = np.array(problem["image"])
        angle = problem["angle"]
        
        rotated_image = scipy.ndimage.rotate(
            image, angle, reshape=False, order=3, mode='constant', cval=0.0
        )
        
        return {"rotated_image": rotated_image}